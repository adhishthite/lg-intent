.PHONY: install format lint check test clean run graph-ascii graph-mermaid pre-commit pre-commit-install

install:
	uv sync

pre-commit-install:
	uv run pre-commit install

pre-commit:
	uv run pre-commit run --all-files
	$(MAKE) clean

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

# Run with custom query: make run QUERY="What's the PTO policy?"
# Run demo queries: make run
QUERY ?=
run:
ifdef QUERY
	uv run python src/main.py "$(QUERY)"
else
	uv run python src/main.py
endif

graph-ascii:
	PYTHONPATH=src uv run python -c "from enterprise_rag.graph import print_graph_ascii; print_graph_ascii()"

graph-mermaid:
	PYTHONPATH=src uv run python -c "from enterprise_rag.graph import print_graph_mermaid; print_graph_mermaid()"
