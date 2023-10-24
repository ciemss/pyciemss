lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

tests: lint FORCE
	pytest -v test -n auto

FORCE: