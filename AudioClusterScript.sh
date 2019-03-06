#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=49000  # memory in Mb
#SBATCH --time=0-8:00:00


export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

echo ${STUDENT_ID}

mkdir -p /disk/scratch/${STUDENT_ID}
export TMPDIR=/disk/scratch/${STUDENT_ID}
export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/datasets
export DATASET_DIR=${TMP}/datasets/
mkdir -p ${TMP}/MLPProjectAudio
export CODE_DIR=${TMP}/MLPProjectAudio/
mkdir -p ${DATASET_DIR}/train
mkdir -p ${DATASET_DIR}/test

if [ $# != 1 ]; then
echo "Usage: sbatch AudioClusterScript.sh <experiment_number>"
exit 1;
fi
number=$1


# Activate the relevant virtual environment:
#source ~/.bashrc
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
##print the name of the GPU BOX where the job is running



##  SYnc data in the headnode  with the job's GPU BOX
rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/data_short/ ${DATASET_DIR}/test
rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/data_short_train/ ${DATASET_DIR}/train
rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/data/ ${DATASET_DIR}

rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/MLPProjectAudio/ /disk/scratch/${STUDENT_ID}/MLPProjectAudio


cd /disk/scratch/${STUDENT_ID}
cd datasets/FSDnoisy18k.meta
ls
pwd
cd ../..
cd MLPProjectAudio
pwd
echo databse directory ${DATASET_DIR}
#bash run_experiment_preprocessing.sh


echo entrando a python
python MLP_CW2/mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 5 --num_filters 12,24,48,48,24 --kernel_size 3 --batch_size 64 --use_gpu True --gpu_id "0,1,2,3" --use_cluster True --num_epochs 20 --training_instances 17310 --val_instances 275 --experiment_name exp_audio_${number}
echo pase python

# recovering data
cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_val.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/
cp /disk/scratch/${STUDENT_ID}/MLPProjectAudio/exp_audio_${number} /home/${STUDENT_ID}/ExperimentsAudio
cp /disk/scratch/s1870525/datasets/experiment_config.txt /home/${STUDENT_ID}/ExpermentsAudio/exp_audio_${number}
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_val.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_train.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_test.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
