install:
	pip uninstall ufldl -y &&\
	pip install -e .

tests:
	pytest -s
