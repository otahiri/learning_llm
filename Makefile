all: run

run:
	uv run python3 main.py

intall:
	uv sync

clean: 
	rm -rf __pycache__ .mypy_cache
