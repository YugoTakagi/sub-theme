# sub-theme
python-code for 副テーマ.

#### Dawnload dataset.
% git clone https://github.com/ouktlab/Hazumi1902









>> How to use tensorflow with gpu. <<
ダメだった．．

以下，ダウンロード方法．
% qsub -I
% module load singularity/3.7.1
> tensorflow.__version__ : 2.1.0, python -V : 3.6.0
% singularity pull docker://nvcr.io/nvidia/tensorflow:20.03-tf2-py3
% ls
% singularity exec --nv /home/$USER/tensorflow_20.03-tf2-py3.sif python
> tensorflowは使えたが，gpuが使えない．．．

ダウンロード後．
% qsub -q GPU-1 -I -lselect=1:ngpus=1
% module load singularity/3.7.1
% singularity exec --nv /home/$USER/tensorflow_20.03-tf2-py3.sif python
>>> import tensorflow as tf
>>> print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
Num GPUs Available:  1
>>>
> tensorflow with gpuが使えた．
> でも，kerasがないって．．．
% singularity exec --nv /home/$USER/tensorflow_20.03-tf2-py3.sif bash
//% pip install keras
% pip install keras==2.3.1
% pip install pandas
% pip install scikit-learn
% pip install matplotlib