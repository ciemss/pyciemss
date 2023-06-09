#!/bin/bash
set -euxo pipefail

isort --profile black src/pyciemss/ test/
black src/pyciemss/ test/
