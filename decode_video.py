import os
import fire
import mmcv
import numpy as np
from copy import deepcopy
from PIL import Image
from moviepy.editor import VideoFileClip


def run(
    video_root="magicdrive-t-log/submission/video",
    gen_root="magicdrive-t-log/evaluation/generated_samples",
    data_info="data/nuscenes_mmdet3d_2/nuscenes_infos_temporal_val_3keyframes.pkl",
    generation_times=4,
):
    view_order = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    ]
    key_frame_index = [0, 6, 12]
    out_size = (1600, 900)

    data = mmcv.load(data_info)

    # we sort infos for sequential read
    token_info_dict = {}
    first_frame_tokens = []
    for info in data['infos']:
        token_info_dict[info['token']] = info
        if info['prev'] == "":
            first_frame_tokens.append(info['token'])
    print(f"loaded scenes: {first_frame_tokens.__len__()}")

    test_infos = []
    for gen_idx in range(generation_times):
        save_dir = os.path.join(gen_root, f"gen_samples_{gen_idx}")

        # make save dir
        for view in view_order:
            cam_save_dir = os.path.join(save_dir, view)
            os.makedirs(cam_save_dir)

        # read video and save image
        for token in first_frame_tokens:
            video_dir = os.path.join(video_root, f"{token}_gen{gen_idx}")

            # make new info for this video frames
            this_kf_info = deepcopy(token_info_dict[token])
            this_kf_info['scene_token'] = this_kf_info['scene_token'] + f"_gen{gen_idx}"
            vid_infos = [this_kf_info]
            for _ in range(len(key_frame_index) - 1):
                this_kf_info = deepcopy(token_info_dict[this_kf_info['next']])
                this_kf_info['scene_token'] = this_kf_info['scene_token'] + f"_gen{gen_idx}"
                vid_infos.append(this_kf_info)

            # save image and update infos
            for view in view_order:
                cam_save_dir = os.path.join(save_dir, view)
                video_name = os.path.join(video_dir, f"{token}_{view}.mp4")
                if not os.path.exists(video_name):
                    print(f"There is no video for {token}")
                    break

                clip = VideoFileClip(video_name)

                # iterating frames
                frames = clip.iter_frames()
                kf_idx = 0
                for idx, frame in enumerate(frames):
                    if idx not in key_frame_index:
                        continue

                    this_kf_info = vid_infos[kf_idx]
                    # save image
                    this_img = Image.fromarray(frame).resize(out_size)
                    img_name = os.path.splitext(os.path.basename(
                        this_kf_info['cams'][view]['data_path']))[0] + ".png"
                    save_path = os.path.join(cam_save_dir, img_name)
                    assert not os.path.exists(save_path)
                    this_img.save(save_path)
                    this_kf_info['cams'][view]['data_path'] = save_path
                    kf_idx += 1
            else:  # if not break, we have saved images and changed infos
                test_infos += vid_infos
    print(f"test infos length = {len(test_infos)}")
    save_path = os.path.join(
        gen_root, "nuscenes_infos_temporal_val_3keyframes.pkl")
    mmcv.dump(
        {"infos": test_infos, "metadata": data['metadata']},
        save_path
    )
    print(f"Please load data info from {save_path} for testing.")


if __name__ == "__main__":
    fire.Fire(run)
