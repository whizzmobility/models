DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

EXPERIMENT="seg_deeplabv3plus_scooter"
CONFIG_FILENAME="deeplabv3plus_resnet50_scooter_gpu"
MODEL_DIR="D:/repos/data_root/${CONFIG_FILENAME}"
IMAGE_DIR_GLOB="D:/models/test_images/11-May-2019-18-31-31/*.png"

set +o posix
exec > >(tee ${MODEL_DIR}/run_model.log) 2>&1
python ../serving/run_model.py \
    --experiment="${EXPERIMENT}" \
    --model_dir="${MODEL_DIR}" \
    --image_path_glob="${IMAGE_DIR_GLOB}" \
    --output_dir="${MODEL_DIR}/runs/trained" \
    --visualise=1 \
    --stitch_original=1 \
    --batch_size=1 \
    --input_image_size=512,512 \
    --config_file="../configs/experiments/semantic_segmentation/${CONFIG_FILENAME}.yaml"
