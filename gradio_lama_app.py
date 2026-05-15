from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from loguru import logger

from project_ai.v3.engine.helper import load_jit_model, norm_img, pad_img_to_modulo, resize_max_size

LAMA_MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_runtime(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return f"CUDA enabled on `{torch.cuda.get_device_name(0)}`"
    if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple Metal (MPS) enabled"
    return "Running on CPU"


def to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def ensure_rgb(image: Any) -> np.ndarray:
    array = to_uint8(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim != 3:
        raise ValueError("Unsupported image shape from Gradio editor.")
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    if array.shape[2] == 3:
        return array
    raise ValueError("Unsupported number of image channels from Gradio editor.")


def ensure_rgba(image: Any) -> np.ndarray:
    array = to_uint8(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGBA)
    if array.ndim != 3:
        raise ValueError("Unsupported image shape from Gradio editor.")
    if array.shape[2] == 4:
        return array
    if array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2RGBA)
    raise ValueError("Unsupported number of image channels from Gradio editor.")


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def extract_editor_image_and_mask(editor_value: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not editor_value:
        raise ValueError("Upload an image before running the inpainting app.")

    background = editor_value.get("background")
    composite = editor_value.get("composite")
    if background is None and composite is None:
        raise ValueError("Upload an image before drawing a mask.")

    base_image = ensure_rgb(background if background is not None else composite)
    height, width = base_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for layer in editor_value.get("layers") or []:
        layer_rgba = resize_image(ensure_rgba(layer), height, width)
        layer_mask = layer_rgba[:, :, 3]
        if not np.any(layer_mask):
            layer_mask = np.max(layer_rgba[:, :, :3], axis=2)
        mask = np.maximum(mask, layer_mask)

    if not np.any(mask) and background is not None and composite is not None:
        background_rgba = resize_image(ensure_rgba(background), height, width)
        composite_rgba = resize_image(ensure_rgba(composite), height, width)
        rgb_delta = np.max(
            np.abs(
                composite_rgba[:, :, :3].astype(np.int16)
                - background_rgba[:, :, :3].astype(np.int16)
            ),
            axis=2,
        ).astype(np.uint8)
        alpha_delta = np.abs(
            composite_rgba[:, :, 3].astype(np.int16)
            - background_rgba[:, :, 3].astype(np.int16)
        ).astype(np.uint8)
        mask = np.maximum(mask, rgb_delta)
        mask = np.maximum(mask, alpha_delta)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return base_image, mask


def expand_mask(mask: np.ndarray, dilation_radius: int) -> np.ndarray:
    if dilation_radius <= 0:
        return mask
    kernel_size = dilation_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(mask, kernel, iterations=1)


def load_initial_image(image_path: Optional[Path]) -> Optional[np.ndarray]:
    if image_path is None:
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class LaMaService:
    def __init__(self, device: str, resize_limit: int) -> None:
        self.device = torch.device(device)
        self.resize_limit = resize_limit
        self.lock = threading.Lock()
        self.model = None

    def ensure_model(self):
        if self.model is None:
            logger.info("Loading LaMa model")
            self.model = load_jit_model(LAMA_MODEL_URL, self.device, LAMA_MODEL_MD5).eval()
        return self.model

    def _run_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        model = self.ensure_model()
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(image, mod=8)
        pad_mask = pad_img_to_modulo(mask, mod=8)

        image_tensor = torch.from_numpy(norm_img(pad_image)).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(norm_img(pad_mask)).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor > 0).to(image_tensor.dtype)

        output = model(image_tensor, mask_tensor)[0].permute(1, 2, 0).detach().cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return output[:origin_height, :origin_width, :]

    def run(
        self,
        editor_value: Optional[dict],
        mask_dilation: int,
        resize_limit: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        try:
            image, mask = extract_editor_image_and_mask(editor_value)
        except Exception as exc:
            return None, None, f"Input error: {exc}"

        mask = expand_mask(mask, mask_dilation)
        if not np.any(mask):
            return None, None, "Draw a scribble mask over the object you want to remove."

        try:
            with self.lock:
                start_time = time.perf_counter()

                if max(image.shape[:2]) > resize_limit:
                    original_size = image.shape[:2]
                    resized_image = resize_max_size(image, size_limit=resize_limit)
                    resized_mask = resize_max_size(
                        mask, size_limit=resize_limit, interpolation=cv2.INTER_NEAREST
                    )
                    result_rgb = self._run_lama(resized_image, resized_mask)
                    result_rgb = cv2.resize(
                        result_rgb,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    keep_pixels = mask < 127
                    result_rgb[keep_pixels] = image[keep_pixels]
                else:
                    result_rgb = self._run_lama(image, mask)
                    blend = mask[:, :, None] / 255.0
                    result_rgb = (
                        result_rgb.astype(np.float32) * blend
                        + image.astype(np.float32) * (1.0 - blend)
                    ).astype(np.uint8)

                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
        except Exception as exc:
            logger.exception("LaMa object removal failed")
            return None, None, f"Inpainting failed: {exc}"

        mask_preview = np.repeat(mask[:, :, None], 3, axis=2)
        status = (
            f"Completed in `{elapsed_ms:.0f} ms` using `lama` on `{self.device.type}` "
            f"for `{image.shape[1]}x{image.shape[0]}`."
        )
        return result_rgb, mask_preview, status


def build_demo(service: LaMaService, initial_image: Optional[np.ndarray]) -> gr.Blocks:
    with gr.Blocks(title="ProjectAI V3 LaMa Object Removal") as demo:
        gr.Markdown(
            f"""
            # AI-Powered Image Inpainting & Object Removal

            This Gradio 6 app uses the ProjectAI V3 LaMa backend for scribble-based object removal.

            **Runtime**: {describe_runtime(service.device.type)}
            """
        )

        with gr.Row():
            with gr.Column(scale=7):
                editor = gr.ImageEditor(
                    value=initial_image,
                    type="numpy",
                    image_mode="RGBA",
                    label="Photo + scribble mask",
                    height=640,
                    sources=["upload", "clipboard"],
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=22),
                    eraser=gr.Eraser(default_size=20),
                    transforms=("crop", "resize"),
                )
            with gr.Column(scale=5):
                mask_dilation = gr.Slider(
                    minimum=0,
                    maximum=64,
                    value=12,
                    step=1,
                    label="Mask expansion (px)",
                )
                resize_limit = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    value=1280,
                    step=64,
                    label="Resize limit",
                )
                with gr.Row():
                    run_button = gr.Button("Remove Object", variant="primary")
                    clear_button = gr.Button("Clear Session")
                status = gr.Markdown(
                    "Paint over the object, then click **Remove Object**."
                )

        with gr.Row():
            result_image = gr.Image(type="numpy", label="Inpainted result", height=420)
            mask_preview = gr.Image(type="numpy", label="Mask used", height=420)

        run_button.click(
            fn=service.run,
            inputs=[editor, mask_dilation, resize_limit],
            outputs=[result_image, mask_preview, status],
            api_name="remove_object",
        )
        clear_button.click(
            fn=lambda: (None, None, None, "Paint over a new object to continue."),
            outputs=[editor, result_image, mask_preview, status],
            queue=False,
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a Gradio 6 app for LaMa-based object removal."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default=detect_default_device(), choices=["cpu", "cuda", "mps"])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    initial_image = load_initial_image(args.input)
    service = LaMaService(device=args.device, resize_limit=1280)
    demo = build_demo(service, initial_image)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=args.inbrowser,
        share=True,
    )


if __name__ == "__main__":
    main()
