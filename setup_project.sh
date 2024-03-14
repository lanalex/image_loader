#!/bin/bash

# ----------------------------------------------------------------------------------------------------
# Parameters for execution
# ----------------------------------------------------------------------------------------------------
FLAG_USE_ENV=false

# ----------------------------------------------------------------------------------------------------
# Use conda environment if requested
# ----------------------------------------------------------------------------------------------------
if [ "$FLAG_USE_ENV" = true ]; then
  # Define the path to the conda installation
  CONDA_PATH="/opt/miniconda3"

  # Define the name of the Conda environment to use
  ENV_NAME="seedoo_geoetl_v01"

  # Source the Conda command line tools
  source "${CONDA_PATH}/etc/profile.d/conda.sh"

  # Ensure we start from the base environment
  conda deactivate

  # Check if the Conda environment exists and activate it
  if conda env list | grep -q "^${ENV_NAME}\s"; then
      echo "Activating environment '${ENV_NAME}'."
      conda activate "$ENV_NAME"
  else
      echo "Environment '${ENV_NAME}' not found. Please create the environment first."
      exit 1
  fi
fi

# ----------------------------------------------------------------------------------------------------
# Setup project
# ----------------------------------------------------------------------------------------------------
# Upgrade 'build' to ensure the latest setuptools is installed, necessary for the build process
pip install --upgrade build

# Install 'pip-tools' for compiling 'requirements.in' into 'requirements.txt'
# In a previous project, we had pip-tools==7.3.0.
pip install pip-tools

# Install project dependencies
pip-compile requirements.in --no-strip-extras
pip install -r requirements.txt
pip install -e ./

# Optionally build the project if we need to distribute it
# python -m build

# ----------------------------------------------------------------------------------------------------
# Example calls
# ----------------------------------------------------------------------------------------------------
# Without arguments:
# ./setup_proj.sh
#