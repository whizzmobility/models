DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

DATA_DIR="D:/data"
DATASET_NAME="SUTD_detector"
SUBSET=""
OUTPUT_PREFIX="detect_env"
echo $DATASET_DIR

set +o posix
exec > >(tee create_yolo_tfrecord_output.log) 2>&1
python ../data/create_yolo_tfrecord.py \
  --image_dir="${DATA_DIR}/${DATASET_NAME}/${SUBSET}" \
  --output_file_prefix="D:/data/whizz_tf/${OUTPUT_PREFIX}_${SUBSET}_${DATASET_NAME}" \
  --num_shards=4