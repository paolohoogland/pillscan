.PHONY: install venv

venv:
	python3 -m venv venv

install:
	pip3 install tensorflow keras scikit-learn pandas numpy matplotlib pillow torch torchvision tqdm
