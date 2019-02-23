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

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
mkdir -p ${TMP}/MLPProjectAudio/
export DATASET_DIR=${TMP}/datasets/
export DATASET_DIR=${TMP}/MLPProjectAudio/

# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
#print the name of the GPU BOX where the job is running
srun hostname
# SYnc data in the headnode  with the job's GPU BOX
rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/data/ /disk/scratch/${STUDENT_ID}/datasets/

rsync -ua --progress /home/${STUDENT_ID}/ExperimentsAudio/MLPProjectAudio/ /disk/scratch/${STUDENT_ID}/MLPProjectAudio


#python train_evaluate_emnist_classification_system.py --batch_size 100 --continue_from_epoch -1 --seed 0 \
 #                                                     --image_num_channels 1 --image_height 28 --image_width 28 \
  #                                                    --dim_reduction_type "strided" --num_layers 4 --num_filters 64 \
   #                                                   --num_epochs 100 --experiment_name 'emnist_test_multi_gpu_exp' \
    #                                                  --use_gpu "True" --gpu_id "0,1,2,3" --weight_decay_coefficient 0. \
     #              #                                   --dataset_name "emnist"
pwd

#python MLPProjectAudio/MLP_CW2/mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_filters 5,5,5 --batch_size 64 --use_gpu True --gpu_id "0,1,2,3" --use_cluster True --num_epochs 1
