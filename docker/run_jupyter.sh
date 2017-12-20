#!/bin/bash

cd ${1:-$HOME}
/home/nanograv/miniconda2/bin/jupyter notebook --no-browser --port 8888 --ip=0.0.0.0
