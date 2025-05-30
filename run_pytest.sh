#!/bin/bash

# Check if pytest is available in the system's PATH
if command -v pytest &>/dev/null; then
    echo "pytest found in PATH. Running system pytest."
    exec pytest "$@"
else
    # If not found, try the virtual environment path
    VENV_PYTEST="./.venv/bin/pytest"
    if [ -f "$VENV_PYTEST" ]; then
        echo "pytest not found in PATH. Running pytest from virtual environment."
        exec "$VENV_PYTEST" "$@"
    else
        echo "Error: pytest executable not found in PATH or ./.venv/bin/pytest"
        exit 1
    fi
fi
