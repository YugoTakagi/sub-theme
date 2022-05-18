#!/bin/bash
#PBS -q SINGLE
#PBS -N sub-theme
#PBS -l select=1
#PBS -j oe

cd ${PBS_O_WORKDIR}

python train_and_test.py
