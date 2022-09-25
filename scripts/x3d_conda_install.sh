#!/bin/bash

CONDA_PATH=$HOME/SOFTWARE/anaconda
FLOWPY_ROOT=$HOME/data/SOFTWARE/flowpy

if [ ! -f ../setup.py ]; then
	echo -e "This script must be run from the \
			 the scripts directory...exiting\n"
	exit 1
fi
echo -e "\n *** SETTING UP x3d_post ***\n"

source ./helper_funcs.sh

cd ../

X3D_POST_ROOT=${PWD}
echo -e "Root directory: $X3D_POST_ROOT"


echo -e "Using conda environment: x3d_post"

source $CONDA_PATH/etc/profile.d/conda.sh 2>/dev/null

test_cmd conda "Check the anaconda root path, conda not found"

conda env list | grep -w x3d_post > /dev/null

if [ $? -eq 0 ]; then
	echo -e "Conda environment already exists."
	CONDA_ENV_EXISTS=1
else
	echo -e "Conda environment doesn't exist. Creating a new one"
	CONDA_ENV_EXISTS=0
fi


if [ $CONDA_ENV_EXISTS -eq 0 ]; then
	conda env create -f scripts/x3d_post.yml
else
	conda env update -f scripts/x3d_post.yml  --prune
fi

conda activate x3d_post

PYBIN=$(which python3)
test_cmd $PYBIN

# install flowpy

$PYBIN $FLOWPY_ROOT/setup.py install

$PYBIN -c "import flowpy" > /dev/null
test_return "flowpy import test failed" 

# install x3d_post
$PYBIN $X3D_POST_ROOT/setup.py install

cd $X3D_POST_ROOT/scripts

$PYBIN -c "import x3d_post" > /dev/null

test_return "x3d_post import test failed" 
conda deactivate

echo -e "### Finished Setup of x3d_post"