install:
	pip install -r requirements.txt

FILE ?= HW3/assignment3.out.ipynb

html:
	black --line-length 96 $(FILE)
	python3 -m jupyter nbconvert --to html --template classic $(FILE)
