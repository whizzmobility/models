. ./environment_setup.sh

CKPT_DIR=D:/models/experiments/ckpt_renaming

set +o posix
exec > >(tee ${CKPT_DIR}/sync_variables.log) 2>&1
python ../serving/sync_variables.py \
    --ckpt_ref="${CKPT_DIR}/actual" \
    --ckpt_target="${CKPT_DIR}/keras"
