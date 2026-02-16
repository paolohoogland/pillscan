.PHONY: install venv api

venv:
	python3 -m venv venv

install:
	pip3 install tensorflow keras scikit-learn pandas numpy matplotlib pillow torch torchvision tqdm

install-api:
	pip3 install fastapi uvicorn python-multipart

api:
	cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000
