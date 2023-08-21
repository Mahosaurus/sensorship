unit_tests:
	echo "Executing Unit Tests..."
	python -m pytest -vv tests/ --random-order --random-order-bucket=module --random-order-seed=4003465229 --cov-report term --cov=src
	rm .coverage.* 2> /dev/null
	rm .coverage 2> /dev/null
	rm -r .pytest_cache 2> /dev/null