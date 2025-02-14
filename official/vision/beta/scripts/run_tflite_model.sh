. ./environment_setup.sh
IMAGE_DIR_GLOB="D:/models/test_images/11-May-2019-18-31-31/*.png"

set +o posix
exec > >(tee ${MODEL_DIR}/run_tflite_model.log) 2>&1
python ../serving/run_tflite_model.py \
    --experiment="${EXPERIMENT}" \
    --batch_size=1 \
    --model_path="${MODEL_DIR}/export/model.tflite" \
    --image_path_glob="${IMAGE_DIR_GLOB}" \
    --output_dir="${MODEL_DIR}/runs/tflite" \
    --visualise=1 \
    --stitch_original=1 \
    --class_names_path='D:/data/SUTD_detector/classes.txt'
