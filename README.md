# W-CODA2024-Track2
This repository is dedicated to Track 2 of the W-CODA 2024 Workshop, "Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving," held at ECCV 2024 in Milano, Italy.

## Test FVD

### Parameter Explanation

1. `base_path`: After running the `test` command of MagicDrive, there will be a `magicdrive-t-log` folder. Please specify the path to this folder. For example, `/MagicDrive/magicdrive-t-log/test`.

2. `data_path`: Please download `nuscenes_infos_temporal_val_12hz.pkl` from HuggingFace and specify the path to it. This file contains the paths to the real frames to be tested.

3. `output_path1` / `output_path2`: The path to cache the generated video tensors / the path to cache the real video tensors. You can choose whether to use `torch.save` to save the processed video tensors, which can help you avoid waiting for frame loading every time you test FVD.

4. `use_cache`: If you have already saved the corresponding tensors, you can choose to directly load them.

### Usage

1. Run the `test` command of MagicDrive to generate the `magicdrive-t-log` folder.

2. Download `nuscenes_infos_temporal_val_12hz.pkl` from HuggingFace and specify the `data_path` parameter.

3. Set the `base_path`, `output_path1`, and `output_path2` parameters according to your setup.

4. If you have previously cached the video tensors, set `use_cache` to `True` to directly load them. Otherwise, set it to `False` to process the frames and optionally save the tensors using `torch.save`.

5. Run the FVD testing script to evaluate the quality of the generated videos compared to the real videos.


## Test Object mAP & Map mIoU

- [Installation](BEVFormer_segmentation_detection/docs/install.md) 
- [Prepare Dataset](BEVFormer_segmentation_detection/docs/prepare_dataset.md)

Eval BEVFormer with 8 GPUs
```
bash test_miou.sh
```