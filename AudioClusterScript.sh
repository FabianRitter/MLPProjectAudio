#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00


export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

echo ${STUDENT_ID}
pwd

mkdir -p /disk/scratch/${STUDENT_ID}
export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
mkdir -p ${TMP}/MLPProjectAudio/
export CODE_DIR=${TMP}/MLPProjectAudio/


# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
##print the name of the GPU BOX where the job is running
srun hostname



##  SYnc data in the headnode  with the job's GPU BOX
rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/data/ /disk/scratch/${STUDENT_ID}/datasets/

rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/MLPProjectAudio/ /disk/scratch/${STUDENT_ID}/MLPProjectAudio

cd /disk/scratch/${STUDENT_ID}

pwd
cd MLPProjectAudio
#bash run_experiment_preprocessing.sh
#mv ../datasets/newpreprocessing/processed_data_eval.hdf5 ../datasets
#mv ../datasets/newpreprocessing/processed_data_test.hdf5 ../datasets
#mv ../datasets/newpreprocessing/processed_data_train.hdf5 ../datasets

python MLP_CW2/mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_filters 24,48,48 --kernel_size 5 --batch_size 64 --use_gpu True --gpu_id "0,1,2,3" --use_cluster True --num_epochs $1


# recovering data

cp /disk/scratch/${STUDENT_ID}/exp_audio/ /home/${STUDENT_ID}/ExperimentsAudio
cp /disk/scratch/${STUDENT_ID}/datasets/exp_audio /home/${STUDENT_ID}/ExperimentsAudio
cp /disk/scratch/${STUDENT_ID}/MLPProjectAudio/exp_audio /home/${STUDENT_ID}/ExperimentsAudio
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_val.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_train.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
#cp /disk/scratch/${STUDENT_ID}/datasets/processed_data_test.hdf5 /home/${STUDENT_ID}/ExperimentsAudio/data/newpreprocessed
