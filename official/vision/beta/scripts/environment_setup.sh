export CUDA_VISIBLE_DEVICES=0,1

# disable shared memory transport and force to use P2P, which is default for NCCL2.6
# not enough memory in /dev/shm otherwise
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=INFO

TASK_TYPE="semantic_segmentation" #"image_classification"
EXPERIMENT="seg_deeplabv3plus_scooter" #"mobilenet_imagenet"
CONFIG_FILENAME="deeplabv3plus_dilatedefficientnetb0_scooter_gpu" #"imagenet_efficientnet_gpu"
MODEL_DIR="/home/whizz/experiments/${CONFIG_FILENAME}"

NUM_GPUS=2
TRAIN_BATCH_SIZE=16 #256
INPUT_PATH="/mnt/ssd2/tfrecords/**" # ""
NUMBER_OF_IMAGES=6915 #Ours - 6915, Imagenet - 1281167
TRAIN_STEPS_PER_EPOCH=$((NUMBER_OF_IMAGES / TRAIN_BATCH_SIZE))
TRAIN_STEPS=$((TRAIN_STEPS_PER_EPOCH * 1000)) # normally 500 epochs
DECAY_STEPS=$((TRAIN_STEPS_PER_EPOCH * 5)) # 2.5
WARMUP_STEPS=$((TRAIN_STEPS_PER_EPOCH * 5)) # 5
SUMMARY_STEPS=$((TRAIN_STEPS_PER_EPOCH * 2))

# 0.001 for segmentation
# 0.008 * TRAIN_BATCH_SIZE / 128 for imagenet classification
INITIAL_LEARNING_RATE=0.016

DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

PARAMS="{ \
  runtime: {num_gpus: ${NUM_GPUS}}, \
  task: { \
    train_data: { \
      global_batch_size: ${TRAIN_BATCH_SIZE}, \
      input_path: ${INPUT_PATH} \
    }, \
    validation_data: { \
      input_path: ${INPUT_PATH} \
    } \
  }, \
  trainer: { \
    optimizer_config: { \
      learning_rate: { exponential: { \
        decay_steps: ${DECAY_STEPS},
        initial_learning_rate: ${INITIAL_LEARNING_RATE} \
      }}, \
      warmup: {linear: {warmup_steps: ${WARMUP_STEPS}}} \
    }, \
    steps_per_loop: ${TRAIN_STEPS_PER_EPOCH}, \
    summary_interval: ${SUMMARY_STEPS}, \
    train_steps: ${TRAIN_STEPS}, \
    validation_interval: ${TRAIN_STEPS_PER_EPOCH}, \
    validation_steps: ${NUMBER_OF_IMAGES}, \
    checkpoint_interval: ${SUMMARY_STEPS}, \
  } \
}"

if [ ! -d $MODEL_DIR ]; then
    mkdir -p ${MODEL_DIR}
fi