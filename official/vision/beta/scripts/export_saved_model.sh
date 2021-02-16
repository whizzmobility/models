. ./environment_setup.sh

set +o posix
exec > >(tee ${MODEL_DIR}/export_saved_model_output.log) 2>&1
python ../serving/export_saved_model.py \
    --experiment="${EXPERIMENT}" \
    --export_dir="${MODEL_DIR}/export" \
    --checkpoint_path="${MODEL_DIR}" \
    --batch_size=1 \
    --input_image_size=256,256 \
    --config_file="../configs/experiments/semantic_segmentation/${CONFIG_FILENAME}.yaml"
