DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

CONFIG_FILENAME="deeplabv3plus_resnet50_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"

set +o posix
exec > >(tee ${MODEL_DIR}/export_tflite_output.log) 2>&1
python ../serving/export_tflite_model.py \
    --saved_model_dir="${MODEL_DIR}/export/saved_model" \
    --export_path="${MODEL_DIR}/export/model.tflite"
