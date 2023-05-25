#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports src/pyciemss/ test/
isort --check --profile black --diff src/pyciemss/ test/
black --check src/pyciemss/ test/
flake8 src/pyciemss/ test/