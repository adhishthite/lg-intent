.PHONY: install format lint check test clean run

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

check: format lint

test:
	uv run python -c "from email_agent import create_email_agent; print('Import successful')"

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache .mypy_cache
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

run:
	uv run python run_agent.py
