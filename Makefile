

all: run


run:
	uv run python3 -m nn_zero_to_hero

debug:

	uv run python3 -m pdb main.py

intall:
	uv sync

clean: 
	rm -rf __pycache__ .mypy_cache
