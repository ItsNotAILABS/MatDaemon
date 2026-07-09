.PHONY: install dev dev-full test platform benchmark benchmark-ai serve mcp docker

install:
	python -m pip install --upgrade pip
	python -m pip install .

dev:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev,api]

dev-full:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev,api,mcp]

test:
	pytest -q

platform:
	matdaemon platform

benchmark:
	python benchmarks/benchmark_suite.py --profile launch --backends numpy tiled --output benchmarks/results

benchmark-ai:
	python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai

serve:
	python -m pip install -e .[api]
	matdaemon serve --host 0.0.0.0 --port 8000

mcp:
	matdaemon mcp

docker:
	docker compose up --build
