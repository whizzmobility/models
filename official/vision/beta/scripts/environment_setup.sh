export CUDA_VISIBLE_DEVICES=0

# disable shared memory transport and force to use P2P, which is default for NCCL2.6
# not enough memory in /dev/shm otherwise
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=INFO

EXPERIMENT="seg_deeplabv3plus_scooter"
CONFIG_FILENAME="deeplabv3plus_dilatedefficientnetb0_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"

NUM_GPUS=2
TRAIN_BATCH_SIZE=16
INPUT_PATH="D:/data/test_data/val**"
NUMBER_OF_IMAGES=6915
TRAIN_STEPS_PER_EPOCH=$((NUMBER_OF_IMAGES / TRAIN_BATCH_SIZE))
TRAIN_STEPS=$((TRAIN_STEPS_PER_EPOCH * 2000)) # normally 500 epochs
DECAY_STEPS=$((TRAIN_STEPS_PER_EPOCH * 24 / 10))
WARMUP_STEPS=$((TRAIN_STEPS_PER_EPOCH * 5))
SUMMARY_STEPS=$((TRAIN_STEPS_PER_EPOCH * 2))

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
      learning_rate: { polynomial: {decay_steps: ${DECAY_STEPS}}}, \
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