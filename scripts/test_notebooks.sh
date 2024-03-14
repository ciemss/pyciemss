#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/*.ipynb"

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace --to notebook $INCLUDED_NOTEBOOKS
CI=1 pytest --nbval-lax $INCLUDED_NOTEBOOKS
