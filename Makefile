.PHONY: run visualize test acceptance check clean

run:
	python main.py run

visualize:
	python main.py visualize

test:
	python -m pytest -q

acceptance:
	python main.py acceptance

check:
	python main.py check

clean:
	rm -rf out .pytest_cache __pycache__ src/rewardfactory/__pycache__ src/rewardfactory/search/__pycache__ tests/__pycache__
