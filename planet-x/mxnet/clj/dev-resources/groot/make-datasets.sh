#!/bin/bash

unset PYTHONPATH

python3 $MXNET_HOME/tools/im2rec.py --list --recursive groot-train train/
python3 $MXNET_HOME/tools/im2rec.py --list --recursive groot-valid valid/
python3 $MXNET_HOME/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 groot-train train/
python3 $MXNET_HOME/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 groot-valid valid/
