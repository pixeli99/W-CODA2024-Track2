#!/bin/bash

INFOS_DIR=${1}
GPUS=${2:-8}

CONFIG="./BEVFormer_segmentation_detection/projects/configs/bevformer/bevformer_base_seg_det_150x150.py"
BEVFORMER_PRETRAIN="../pretrained/bevformer/bevformer_base_seg_det_150.pth"
TIMESTAMP=`date +"%Y-%m-%d_%H-%M-%S"`

CONFIG_NAME=${CONFIG##*/}
LOG_BASE="magicdrive-t-log/evaluation/"${CONFIG_NAME%%.*}/${TIMESTAMP}

echo "Find your results at" ${LOG_BASE}

set -e
set -x
  
# Run the test script
for INFOS in ${INFOS_DIR}/*.pkl; do # Whitespace-safe but not recursive.
  SUB=${INFOS##*/}
  BEVFormer_segmentation_detection/tools/dist_test.sh ${CONFIG} \
    ${BEVFORMER_PRETRAIN} ${GPUS} --cfg-options data.test.ann_file=${INFOS} \
    --log_timestamp ${TIMESTAMP} --log_part ${SUB}
done

echo "Reading results from ${LOG_BASE}"
python read_map_miou.py ${LOG_BASE}

