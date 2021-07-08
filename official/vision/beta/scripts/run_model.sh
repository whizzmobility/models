. ./environment_setup.sh
IMAGE_DIR_GLOB="D:/models/test_images/11-May-2019-18-31-31/*.png"

set +o posix
exec > >(tee ${MODEL_DIR}/run_model.log) 2>&1
python ../serving/run_model.py \
    --experiment="${EXPERIMENT}" \
    --batch_size=1 \
    --model_dir="${MODEL_DIR}" \
    --image_path_glob="${IMAGE_DIR_GLOB}" \
    --output_dir="${MODEL_DIR}/runs/trained" \
    --visualise=1 \
    --stitch_original=1
