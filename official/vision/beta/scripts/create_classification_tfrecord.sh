DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

DATA_DIR="D:/data"
DATASET_NAME="detector"
DATASET_DIR="${DATA_DIR}/${DATASET_NAME}"
INNER_DIR_PREFIX="detector"
SUBSET="val"
echo $DATASET_DIR

set +o posix
exec > >(tee create_img_tf_record_output.log) 2>&1
python ../data/create_classification_tfrecord.py \
  --image_dir="${DATASET_DIR}/${INNER_DIR_PREFIX}_images/${SUBSET}" \
  --classes_json="${DATASET_DIR}/classes.json" \
  --output_file_prefix="D:/data/whizz_tf/${DATASET_NAME}_${SUBSET}" \
  --json_key="path_type" \
  --num_shards=8