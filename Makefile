# Makefile for concert video editor project

.PHONY: all clean install run train evaluate

all: install

install:
	pip install -r requirements.txt

run:
	python src/main.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

clean:
	find . -type __pycache__ -exec rm -r {} +