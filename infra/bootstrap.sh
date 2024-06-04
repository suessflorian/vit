#!/bin/bash

python --version

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install pipenv

pipenv requirements > requirements.txt
pip install --ignore-installed -r requirements.txt
