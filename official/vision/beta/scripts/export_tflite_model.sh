. ./environment_setup.sh

set +o posix
exec > >(tee ${MODEL_DIR}/export_tflite_output.log) 2>&1
python ../serving/export_tflite_model.py \
    --saved_model_dir="${MODEL_DIR}/export/saved_model" \
    --export_path="${MODEL_DIR}/export/model.tflite"

python ../serving/export_tflite_model.py \
    --saved_model_dir="${MODEL_DIR}/export_withargmax/saved_model" \
    --export_path="${MODEL_DIR}/export_withargmax/model.tflite"