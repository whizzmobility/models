DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

CONFIG_FILENAME="deeplabv3plus_resnet101_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"
IMAGE_DIR_GLOB="D:/models/test_images/11-May-2019-18-31-31/*.png"

set +o posix
exec > >(tee ${MODEL_DIR}/run_model_output.log) 2>&1
python ../serving/run_saved_model.py \
    --saved_model_dir="${MODEL_DIR}/export/saved_model" \
    --image_dir_glob="${IMAGE_DIR_GLOB}" \
    --output_dir="${MODEL_DIR}/runs/saved_model" \
    --visualise=1
