#!/usr/bin/env bash

# compute coverage of source code by the testing scripts
coverage run --source=src/ -m pytest -v

# report code coverage, including missing lines
coverage report -m
