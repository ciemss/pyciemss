#!/bin/bash
set -euxo pipefail

isort --profile black pyciemss/ test/
black pyciemss/ test/
