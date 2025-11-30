.PHONY: all prepare sweep evaluate

all: prepare sweep evaluate

prepare:
	@echo "Preparing data"
	python3 prepare_data.py
	python3 use_given_clusters.py

sweep:
	@echo "Running alpha_sweep.py"
	python3 alpha_sweep.py

evaluate:
	@echo "Running evaluate_experiments.py"
	python3 evaluate_experiments.py
