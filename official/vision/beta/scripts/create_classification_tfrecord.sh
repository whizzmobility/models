DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

DATA_DIR="/mnt/ssd2"
DATASET_NAME="detector"
INNER_DIR_PREFIX="detector"
SUBSET="val"
OUTPUT_PREFIX="cls_env"
echo $DATASET_DIR

set +o posix
exec > >(tee create_img_tf_record_output.log) 2>&1
python ../data/create_classification_tfrecord.py \
  --image_dir="${DATA_DIR}/${DATASET_NAME}/${INNER_DIR_PREFIX}_images/${SUBSET}" \
  --classes_json="${DATA_DIR}/${DATASET_NAME}/classes.json" \
  --output_file_prefix="/mnt/ssd2/tfrecords/${OUTPUT_PREFIX}_${SUBSET}_${DATASET_NAME}" \
  --json_key="path_type" \
  --num_shards=4