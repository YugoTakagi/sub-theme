#!/bin/bash
#PBS -q GPU-1
#PBS -N tensorflow
#PBS -l select=1
#PBS -j oe

cd ${PBS_O_WORKDIR}
singularity exec --nv /work/opt/container_images/tensorflow_20.03-tf2-py3.sif python train_and_test.py
