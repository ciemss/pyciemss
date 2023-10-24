#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports pyciemss/
isort --check --profile black --diff pyciemss/ test/
black --check pyciemss/ test/
flake8 pyciemss/ test/