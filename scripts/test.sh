#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s -n auto --cov=pyciemss/ --cov=tests --cov-report=term-missing ${@-} --cov-report html
