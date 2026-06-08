.PHONY: install lint format pipeline clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

pipeline:
	uv run python pipeline/00_fetch_deities.py
	uv run python pipeline/01_embed.py
	uv run python pipeline/02_reduce_umap.py
	uv run python pipeline/03_label_topics.py
	uv run python pipeline/04_visualize.py
	uv run python pipeline/05_evaluate.py

clean:
	rm -f data/*.npz data/*.parquet data/*.joblib data/*.html data/*.json
