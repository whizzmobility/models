DATA_ENGINE_FOLDER=$(dirname $(dirname $(dirname $(dirname `pwd`))))
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=${DATA_ENGINE_FOLDER}
else
    export PYTHONPATH=$PYTHONPATH:${DATA_ENGINE_FOLDER}
fi

DATA_DIR="/mnt/ssd2"
DATASET_NAME="scooter_labelled"
INNER_DIR_PREFIX="scooter"
SUBSET="train"
OUTPUT_PREFIX="seg"
echo $DATASET_DIR

set +o posix
exec > >(tee create_img_tf_record_output.log) 2>&1
python ../data/create_img_tf_record.py \
  --image_dir="${DATA_DIR}/${DATASET_NAME}/${INNER_DIR_PREFIX}_images/${SUBSET}" \
  --seg_dir="${DATA_DIR}/${DATASET_NAME}/${INNER_DIR_PREFIX}_seg/${SUBSET}" \
  --output_file_prefix="/mnt/ssd2/tfrecords/${OUTPUT_PREFIX}_${SUBSET}_${DATASET_NAME}" \
  --num_shards=4