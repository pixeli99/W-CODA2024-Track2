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

### Installation
Please refer to the [Installation Guide](BEVFormer_segmentation_detection/docs/install.md) for detailed instructions on how to install the necessary dependencies and set up the environment.

### Prepare Dataset
Follow the instructions in the [Prepare Dataset Guide](BEVFormer_segmentation_detection/docs/prepare_dataset.md) to prepare the dataset required for training and evaluation.

### Process Generated Samples
Use the `process_split_rename_images.py` script to process the generated samples. This script splits the original images into six views (corresponding to the camera views) and renames the resulting images to match the expected file naming convention. It supports processing multiple source folders and multiple generations of samples.

Run the script with the following command:

```bash
python process_split_rename_images.py
```

#### Parameter Explanation

1. `base_folder`: The base directory containing the source folders with image frames to be processed. Example: `/MagicDrive/magicdrive-t-log/test`.

2. `folder_prefix`: The prefix of the source folders within the base folder. Each source folder will have this prefix followed by an index number. Example: `SDv1.5mv-rawbox-t_2024-06-06_10-22_`.

3. `target_folder`: The directory where the processed images will be saved. The script will create subdirectories within this target folder for each generation. Example: `gen_samples`.

4. `info_file`: The path to a data file containing metadata about the images. This file is used to retrieve the original filenames for renaming the cropped images. Example: `/BEVFormer_segmentation_detection/data/nuscenes/nuscenes_infos_temporal_val_3keyframes.pkl`.

5. `view_order`: A list specifying the order of camera views. The images will be split and renamed according to this order. Example: `["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]`.

6. `folder_count`: The number of source folders to be processed. Each folder should have images corresponding to the specified prefix and index. Example: `7`.

7. `gen_count`: The number of generations for which the images will be processed. The script will create separate target directories for each generation. Example: `4`.

### Run mIoU/mAP Test

To perform the mIoU test with 8 GPUs, use the test_miou.sh script. This script will:

1.	Check and remove the existing symbolic link ./data/nuscenes/samples if it exists.
2.	Create a new symbolic link ./data/nuscenes/samples pointing to the target folder (e.g., “/W-CODA2024-Track2/gen_samples_0”).
3.	Run the ./tools/dist_test.sh script to perform the mIoU test on the BEVFormer model using the specified configuration file and checkpoint.

The script is designed to loop through a list of target folders (e.g., “gen_samples_0”, “gen_samples_1”, “gen_samples_2”, “gen_samples_3”), performing the mIoU test on each folder.

Run the script with the following command:
```bash
bash test_miou.sh
```