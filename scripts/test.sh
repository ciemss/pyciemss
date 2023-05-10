#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s --cov=src/pyciemss/ --cov=test --cov-report=term-missing ${@-} --cov-report html
