.PHONY: test demo clean

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 scripts/demo.py
	python3 -m blackbox verify data/demo.jsonl
	python3 -m blackbox replay data/demo.jsonl

clean:
	rm -rf data __pycache__ blackbox/__pycache__ tests/__pycache__ scripts/__pycache__
	find . -name "*.pyc" -delete
