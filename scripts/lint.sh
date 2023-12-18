#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports pyciemss/
isort --check --profile black --diff pyciemss/ tests/
black --check pyciemss/ tests/
flake8 pyciemss/ tests/