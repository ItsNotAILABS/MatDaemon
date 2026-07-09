.PHONY: install dev test benchmark benchmark-ai serve docker

install:
	python -m pip install --upgrade pip
	python -m pip install .

dev:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

test:
	pytest -q

benchmark:
	python benchmarks/benchmark_suite.py --profile launch --backends numpy tiled --output benchmarks/results

benchmark-ai:
	python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai

serve:
	python -m pip install -e .[api]
	matdaemon serve --host 0.0.0.0 --port 8000

docker:
	docker compose up --build
