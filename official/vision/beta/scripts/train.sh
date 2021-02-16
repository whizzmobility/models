DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

EXPERIMENT="seg_deeplabv3plus_scooter"
CONFIG_FILENAME="deeplabv3plus_efficientnetb0_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"

if [ ! -d $MODEL_DIR ]; then
    mkdir -p ${MODEL_DIR}
fi

# mode train_and_eval does not work yet. error at keras metric for eval 
# ValueError: SyncOnReadVariable does not support `assign_add` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`
# See https://github.com/tensorflow/models/issues/8588

set +o posix
exec > >(tee ${MODEL_DIR}/train_output.log) 2>&1
python ../train.py \
    --experiment="${EXPERIMENT}" \
    --mode="train" \
    --model_dir="${MODEL_DIR}" \
    --config_file="../configs/experiments/semantic_segmentation/${CONFIG_FILENAME}.yaml"
