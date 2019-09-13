init:
	pip install -r requirements.txt

test:
	python -m pytest --junitxml=tests/unit_test_outputs.xml tests

