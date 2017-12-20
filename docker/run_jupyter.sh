#!/bin/bash

cd ${1:-$HOME}
/home/nanograv/.local/miniconda/bin/jupyter notebook --no-browser --port 8888 --ip=0.0.0.0
