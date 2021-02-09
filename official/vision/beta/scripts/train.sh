export PYTHONPATH="D:/repos/models/"

MODEL_DIR=D:/repos/data_root/test_data

# mode train_and_eval does not work yet. error at keras metric for eval 
# ValueError: SyncOnReadVariable does not support `assign_add` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`
# See https://github.com/tensorflow/models/issues/8588

# TODO: modify strategy type, modify batch size, use multigpu

python ../train.py \
    --experiment="seg_deeplabv3plus_scooter" \
    --mode="train" \
    --model_dir="${MODEL_DIR}"
    # 1> "${MODEL_DIR}/output.log" 2> "${MODEL_DIR}/error.log"
