#!/bin/tcsh
#PBS -q SINGLE
#PBS -N sub-theme
#PBS -l select=1
#PBS -j oe

# for conda
setenv PATH /home/$USER/.conda/bin:$PATH
source /home/$USER/.conda/etc/profile.d/conda.csh

#source /home/$USER/.cshrc

conda init tcsh
conda activate Tensorflow_env

cd ${PBS_O_WORKDIR}

python train_and_test.py
