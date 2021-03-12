. ./environment_setup.sh

# mode train_and_eval does not work yet. error at keras metric for eval 
# ValueError: SyncOnReadVariable does not support `assign_add` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`
# See https://github.com/tensorflow/models/issues/8588

set +o posix
exec > >(tee ${MODEL_DIR}/train_output.log) 2>&1
python ../train.py \
    --experiment="${EXPERIMENT}" \
    --mode="train" \
    --model_dir="${MODEL_DIR}" \
    --config_file="../configs/experiments/${TASK_TYPE}/${CONFIG_FILENAME}.yaml" \
    --params_override="${PARAMS}"
