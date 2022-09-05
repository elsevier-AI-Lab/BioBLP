# Self-Documented Makefile https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: clean setup install

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL=/bin/bash
PYTHON = python
PROJECT_NAME = bioblp
PACKAGE_NAME = bioblp
PYTHON_INTERPRETER = python3
KERNEL_NAME=Python (${PROJECT_NAME})
PYTHON_FULL_V = $(shell python -V)
PYTHON_V := $(PYTHON_FULL_V:Python%=%)
CONDA_ENV=${PROJECT_NAME}-env
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
#PYTHON_V=3.8.6

#################################################################################
# COMMANDS                                                                      #
#################################################################################

default: help

print-%: ## Prints a variable value. Usage: make print-VARIABLE, eg: make print-TAG, result: TAG = 0.0.0
	@echo $* = $($*)

setup:
	make install_poetry
	@echo $(shell poetry --version) || "Install Poetry"

install_poetry:  ## installs poetry. Remember to `source /home/jovyan/.poetry/env` from a terminal after running this recipe. Need only be run once
	curl -sSL https://install.python-poetry.org | python3 -

install:
	poetry install
	poetry export -f requirements.txt --without-hashes  --dev --output requirements.txt


update:
	poetry update
	poetry export -f requirements.txt --without-hashes  --dev --output requirements.txt

test:
	make lint
	poetry run pytest tests

create_ipython_kernel:
	poetry run ipython kernel install --user --display-name="${KERNEL_NAME}"

freeze_requirements: ## Writes python project dependencies as a requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

freeze_dev_requirements: ## Writes python project dependencies (including dev) as a requirements-dev.txt
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --dev

dist: ## Builds a distribution package with version ${PACKAGE_NAME}.__version__, eg: dist/test_me-0.0.0.tar.gz
	make clean
	poetry build


### JH setup

setup_jh_env:
	make conda_setup
	make create_conda_env
	make create_conda_kernel

conda_setup: # ensures conda env is persistent, need run only once
	mkdir -p /home/jovyan/.conda/pkgs/
	touch /home/jovyan/.conda/pkgs/urls.txt

create_conda_env:
	conda create --yes --prefix /home/jovyan/.conda/envs/${CONDA_ENV} ipykernel
	#conda create --yes --prefix /home/jovyan/.conda/envs/${CONDA_ENV} python==${PYTHON_V} ipykernel
	($(CONDA_ACTIVATE) /home/jovyan/.conda/envs/${CONDA_ENV} | make setup | source /home/jovyan/.poetry/env)
	# to install the project module as a dependency
	($(CONDA_ACTIVATE) /home/jovyan/.conda/envs/${CONDA_ENV} | make install)
	conda env export -n ${CONDA_ENV} -f ${PROJECT_DIR}/environment.yml

create_conda_kernel:
	python -m ipykernel install --user --name=${CONDA_ENV} --display-name="${KERNEL_NAME}"

update_conda_env:
	#($(CONDA_ACTIVATE) /home/jovyan/.conda/envs/${CONDA_ENV} | make update)
	conda env update --name ${CONDA_ENV} -f ${PROJECT_DIR}/environment.yml  --prune
