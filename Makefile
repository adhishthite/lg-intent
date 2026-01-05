.PHONY: install format lint check test clean run graph-ascii graph-mermaid

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

check: format lint

test:
	uv run pytest -v

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache .mypy_cache
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

run:
	uv run python src/main.py

graph-ascii:
	PYTHONPATH=src uv run python -c "from enterprise_rag.graph import print_graph_ascii; print_graph_ascii()"

graph-mermaid:
	PYTHONPATH=src uv run python -c "from enterprise_rag.graph import print_graph_mermaid; print_graph_mermaid()"
