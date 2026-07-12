.PHONY: install test lint preprocess-bgl preprocess-hdfs
install:
	python -m pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src scripts tests
preprocess-bgl:
	python scripts/preprocess.py --config configs/data/bgl.yaml
preprocess-hdfs:
	python scripts/preprocess.py --config configs/data/hdfs.yaml
