export PYTHONPATH="D:/repos/models/"

python ../data/create_img_tf_record.py \
  --image_dir="D:/data/scooter_labelled_small/scooter_images" \
  --seg_dir="D:/data/scooter_labelled_small/scooter_seg" \
  --output_file_prefix="D:/data/test_data/wtf" \
  1> output.log 2> error.log