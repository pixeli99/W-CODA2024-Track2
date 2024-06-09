#!/bin/bash

# Set the list of folders to process
folders=(
  "gen_samples_0"
  "gen_samples_1"
  "gen_samples_2" 
  "gen_samples_3"
)

# Change to the BEVFormer_segmentation_detection directory
cd BEVFormer_segmentation_detection

# Loop through each folder
for folder in "${folders[@]}"
do
  # Check if the symbolic link already exists
  if [ -L "./data/nuscenes/samples" ]; then
    # If it exists, remove the existing symbolic link first
    rm "./data/nuscenes/samples"
  fi
  
  # Create a new symbolic link
  ln -s "/W-CODA2024-Track2/$folder" "./data/nuscenes/samples"
  
  # Run the test script
  ./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base_seg_det_150x150.py ./ckpts/bevformer_base_seg_det_150.pth 8
done
