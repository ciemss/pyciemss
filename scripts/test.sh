#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s -n auto --cov=pyciemss/ --cov=test --cov-report=term-missing ${@-} --cov-report html
