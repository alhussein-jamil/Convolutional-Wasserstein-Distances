.PHONY: install webapp train test test_visualize dataset browse_dataset tensorboard format
PYTHON := python3.10
export ROOT_DIR=$(shell pwd)
export MMDIR=$(HOME)
export PYTHONPATH=$(ROOT_DIR)/training_pipeline_mmlab

## Install environment
install:
	@echo ">> Delete previous venv"
	@rm -rf venv/
	@echo ">> Create venv"
	@$(PYTHON) -m venv venv
	@#./venv/bin/python -m pip install -U pip
	@echo ">> Installing PyTorch"
	@./venv/bin/python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-build-isolation
	@echo ">> Installing dependencies"
	@./venv/bin/python -m pip install -r requirements.txt
	@echo ">> Installing MMCV"
	@./venv/bin/python -m pip install -U openmim
	@./venv/bin/python -m mim install mmengine
	@./venv/bin/python -m mim install "mmcv>=2.0.0rc4, <2.2.0"
	@echo ">> Installing MMDetection"
	@./venv/bin/python -m pip install -v -e $(MMDIR)/mmdetection
	@echo ">> Installing MMPretrain"
	@./venv/bin/python -m mim install -e $(MMDIR)/mmpretrain

## Run webapp
webapp: 
	@./venv/bin/python -m training_pipeline_mmlab.webapp

## Train
train:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=train

## Test
test:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=test

## Visualize test
test_visualize:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=test_visualize

## Create dataset
dataset:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=dataset

## Tensorboard
tensorboard:
	@./venv/bin/python -m tensorboard.main --logdir ./exps

## Visualize dataset
browse_dataset:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=browse_dataset

## Dump training results
dump:
	@./venv/bin/python -m training_pipeline_mmlab.main ++mode=dump


## Run the training pipeline on the gcp vm instance
remote:
	@./venv/bin/python -m training_pipeline_mmlab.main_remote

## Format files with ruff
format:
	./venv/bin/python -m ruff format .  || exit 0
	./venv/bin/python -m ruff check . --fix --exit-zero

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
