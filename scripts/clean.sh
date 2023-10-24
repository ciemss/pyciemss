#!/bin/bash
set -euxo pipefail

isort --profile black pyciemss/ tests/
black pyciemss/ tests/
