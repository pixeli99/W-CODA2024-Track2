# W-CODA2024 Track2: Corner Case Scene Generation
This repository is dedicated to the evaluation of W-CODA2024 Challenge Track2, from the "Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving" workshop held at ECCV 2024.

## How to Submit?

Please check [our website](https://coda-dataset.github.io/w-coda2024/track2/) for registration and detailed guideline.

Take our baseline as an example. To generate the submission, first, run video generation through:

```bash
cd ${MagicDrive_root}
python workshop/test_submit.py \
	resume_from_checkpoint=magicdrive-t-log/links/with_sweeps/SDv1.5mv-rawbox-t_2023-12-04_17-51_2.0t_0.4.3/weight-E4-S77040/ \
	task_id=track2 ++runner.validation_index=all \
    ++dataset.data.val.ann_file=data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval.pkl \
    show_box=false runner.validation_times=4
```

You will get the results at path like `magicdrive-t-log/submission/SDv1.5mv-rawbox-t_2024-06-10_11-08_track2` with content like:

```
├── frames/  # this is the one we need
├── *.mp4
└── ...
```

Then run 

```bash
# this command run for one ${token}
cd ${MagicDrive_root}
python workshop/make_video_from_imgs.py \
	magicdrive-t-log/submission/SDv1.5mv-rawbox-t_2024-06-10_11-08_track2/frames \
    ${TOKEN} \
	--subfix "["_gen0", "_gen1", "_gen2", "_gen3"]"

# this command run for all tokens for submission (150x4=600 videos)
cd ${MagicDrive_root}
python workshop/make_video_from_imgs.py \
	magicdrive-t-log/submission/SDv1.5mv-rawbox-t_2024-06-10_11-08_track2/frames \
    all --subfix "["_gen0", "_gen1", "_gen2", "_gen3"]"
```

The video for submission will be located in `magicdrive-t-log/submission/video`. Now, you can rename the folder `video` to `${teamname}_standard` and submit.

## Evaluation Guidelines

The evaluation metrics used in this codebase are designed to assess the performance of submissions for the ECCV2024 Workshop. Below are the detailed guidelines and requirements for participants:

### Evaluation Metrics
- `mAP` (Mean Average Precision from 3D object detection task)
- `mIoU` (Mean Intersection over Union for BEV segmentation task)
- `FVD` (Fréchet Video Distance)

> **Note:**
>
> - We use [BEVFormer](https://github.com/Bin-ze/BEVFormer_segmentation_detection) for `mAP` and `mIoU` evaluation.
> - Because BEVFormer utilizes key frame information, we only evaluate the detection and segmentation results on the 3 key frames within the 16-frame sequences.
> - Our code is based on the results in submission format. Please be prepared before proceed.

### Get Started

Clone this repo

```bash
git clone --single-branch https://github.com/pixeli99/W-CODA2024-Track2.git
```

Setup python environment and pre-trained weights.

- Detailed instruction for BEVFormer is located in [BEVFormer_segmentation_detection/README.MD](BEVFormer_segmentation_detection/README.MD). We have adapted the code to re-use the environment for MagicDrive. Please do NOT follow the requirements of original repo.
- Download [i3d_pretrained_400.pt](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI) and put it in `${ROOT}/../pretrained/fvd/videogpt`

Add soft-link to data and magicdrive-t-log (please refer to MagicDrive for the content).

```bash
cd ${ROOT}
ln -s /PATH/TO/data
ln -s /PATH/TO/magicdrive-t-log
```

Make sure you have `data/nuscenes_mmdet3d_2/nuscenes_infos_temporal_val_3keyframes.pkl`. If not, please download from the homepage of track2.

### Test FVD

Command:

```bash
cd ${ROOT}
# step 1: generate features for original data
python get_fvd_features_nusc.py \
	--data_info data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval.pkl \
	--out_dir magicdrive-t-log/evaluation/fvd

# step 2: generate features for generated data
python get_fvd_features_gen.py \
	--vid_root magicdrive-t-log/submission/video \
    --out_dir magicdrive-t-log/evaluation/fvd

# step 3: fvd calculation
python fvd_from_npy.py \
	magicdrive-t-log/evaluation/fvd/fvd_feats_ori.npy \
	magicdrive-t-log/evaluation/fvd/fvd_feats_gen.npy 
```

### Test mAP & mIoU

Before testing, we need to decode the video into image format and re-build the meta infos.

```bash
cd ${ROOT}
python decode_video.py \
	--video_root magicdrive-t-log/submission/video \
    --gen_root magicdrive-t-log/evaluation/generated_samples \
    --data_info data/nuscenes_mmdet3d_2/nuscenes_infos_temporal_val_3keyframes.pkl
```

You will have the results like:

```bash
magicdrive-t-log/evaluation/generated_samples/
├── gen_samples_0  # the structure is the same as `nuscenes/samples`
│   ├── CAM_BACK/
│   └── ...
├── gen_samples_1/
├── gen_samples_2/
├── gen_samples_3/
├── nuscenes_infos_temporal_val_3keyframes_gen0.pkl
├── nuscenes_infos_temporal_val_3keyframes_gen1.pkl
├── nuscenes_infos_temporal_val_3keyframes_gen2.pkl
└── nuscenes_infos_temporal_val_3keyframes_gen3.pkl
```

Now you can run BEVFormer with

```bash
cd ${ROOT}
bash test_miou.sh \
	magicdrive-t-log/evaluation/generated_samples 8
# you can use as many gpus as you wish
```

The script will launch evaluation using each of the data infos, and calculate the average after all your data is evaluated. The results will be shown in the terminal. You can also find the results in log and average them:

-  `magicdrive-t-log/evaluation/bevformer_base_seg_det_150x150/*/segmentation_result.json` for `mIoU`
- `magicdrive-t-log/evaluation/bevformer_base_seg_det_150x150/*/pts_bbox/metrics_summary.json` for `mean_ap` as the `mAP`.

