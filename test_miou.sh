#!/bin/bash

INFOS=${1}
GPUS=${2:-8}
  
# Run the test script
BEVFormer_segmentation_detection/tools/dist_test.sh \
  ./BEVFormer_segmentation_detection/projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
  ../pretrained/bevformer/bevformer_base_seg_det_150.pth ${GPUS} \
  --cfg-options data.test.ann_file=${INFOS}
