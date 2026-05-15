# ProjectAI V3

ProjectAI V3 is a locally vendored and rebranded inpainting codebase adapted
into the `project_ai.v3` package layout for this repository.

## Structure

- `engine/`: backend package code, models, plugins, API, CLI, tests
- `web_app/`: frontend source
- `gradio_app.py`: simple object-removal Gradio launcher
- `gradio_lama_app.py`: LaMa-focused Gradio launcher
- `main.py`: CLI entrypoint wrapper

## Usage

Run the server from the repository root:

```bash
python -m project_ai.v3.main start --model lama --device cpu --port 8080
```

Run the Gradio object-removal UI:

```bash
python -m project_ai.v3.gradio_app
```

## Note

This folder preserves the upstream Apache-2.0 `LICENSE` file from the copied
project. Keep that license with this vendored code.
