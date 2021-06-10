export CUDA_VISIBLE_DEVICES=0,1

# disable shared memory transport and force to use P2P, which is default for NCCL2.6
# not enough memory in /dev/shm otherwise
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=INFO

# scooter_labelled: 4251 train, 603 val, 4854 total
# 080420_scooter_halflabelled: 1643 train, 418 val, 2061 total
# imagenet: 1281167 train, 50000 val, 1331167 total
# detector: 2486 train, 621 val
TASK_TYPE="multitask" #"image_classification, semantic_segmentation, multitask"
EXPERIMENT="multitask_vision" #"mobilenet_imagenet, seg_deeplabv3plus_scooter, multitask_vision"
CONFIG_FILENAME="multitask_hardnet_gpu"
MODEL_DIR="/home/whizz/experiments/${CONFIG_FILENAME}"

DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

if [ ! -d $MODEL_DIR ]; then
    mkdir -p ${MODEL_DIR}
fi