DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

EXPERIMENT="seg_deeplabv3plus_scooter"
CONFIG_FILENAME="deeplabv3plus_resnet101_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"

set +o posix
exec > >(tee ${MODEL_DIR}/export_saved_model_output.log) 2>&1
python ../serving/export_saved_model.py \
    --experiment="${EXPERIMENT}" \
    --export_dir="${MODEL_DIR}/export" \
    --checkpoint_path="${MODEL_DIR}" \
    --batch_size=1 \
    --input_image_size=256,256
    # --config_file="../configs/experiments/semantic_segmentation/${CONFIG_FILENAME}.yaml"
