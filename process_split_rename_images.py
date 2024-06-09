import os
import argparse
from PIL import Image
import mmcv
from tqdm import tqdm

def create_dirs(target_folder, view_order):
    """Create target folder and subfolders (if not exist)"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for view in view_order:
        view_dir = os.path.join(target_folder, view)
        if not os.path.exists(view_dir):
            os.makedirs(view_dir)

def process_scene(gen_idx, source_folder, info_idx, scene_idx, data, view_order, target_folder, scene_count):
    """Process a single scene, split images and rename"""
    keyframes = [0, 6, 12]  # Key frame indices
    infos = data['infos']

    for frame_idx in keyframes:
        image_file = f"{scene_idx+1}_{frame_idx}_gen{gen_idx}_{scene_idx*16+frame_idx}.png"
        image_path = os.path.join(source_folder, "frames", image_file)
        if not os.path.exists(image_path):
            print(f"File {image_path} does not exist, skip")
            continue

        image = Image.open(image_path)
        width, height = image.size

        # Ensure image width can be split into six parts
        assert width == 2400 and height == 224, "Image size does not meet expectations"

        for i, view in enumerate(view_order):
            box = (i * 400, 0, (i + 1) * 400, height)
            cropped_image = image.crop(box)

            info_index = (info_idx + scene_idx) * 3 + keyframes.index(frame_idx)
            file_name = os.path.basename(infos[info_index]['cams'][view]['data_path'])
            target_path = os.path.join(target_folder, view, file_name)
            cropped_image = cropped_image.resize((1600, 900))
            cropped_image.save(target_path)

def get_number_of_scenes(folder_path):
        scenes = set()
        frames_path = os.path.join(folder_path, "frames")
        if not os.path.isdir(frames_path):
            return 0
        for filename in os.listdir(frames_path):
            if filename.endswith(".png"):
                scene_num = int(filename.split('_')[0])
                scenes.add(scene_num)
        return len(scenes)

def process_folder(gen_idx, folder_name, info_idx, base_folder, data, view_order, target_folder):
    """Process a single folder"""
    source_folder = os.path.join(base_folder, folder_name)
    scene_count = get_number_of_scenes(source_folder)
    print(source_folder, scene_count, info_idx)
    for scene_idx in tqdm(range(scene_count), desc=f"Processing {folder_name}"):
        process_scene(gen_idx, source_folder, info_idx, scene_idx, data, view_order, target_folder, scene_count)
    return scene_count

def main(args):
    # Load data file
    data = mmcv.load(args.info_file)

    for gen_idx  in range(args.gen_count):
        create_dirs(args.target_folder + f'_{gen_idx}', args.view_order)
        info_idx = 0
        for i in range(1, args.folder_count + 1):
            folder_name = f"{args.folder_prefix}{i}"
            scene_count = process_folder(gen_idx, folder_name, info_idx, args.base_folder, data, args.view_order, args.target_folder + f'_{gen_idx}')
            info_idx += scene_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and rename images")
    parser.add_argument("--base_folder", type=str, default="/MagicDrive/magicdrive-t-log/test", help="Base folder path")
    parser.add_argument("--folder_prefix", type=str, default="SDv1.5mv-rawbox-t_2024-06-06_10-22_", help="Prefix of source folders")
    parser.add_argument("--target_folder", type=str, default="gen_samples", help="Target folder path")
    parser.add_argument("--info_file", type=str, default="/BEVFormer_segmentation_detection/data/nuscenes/nuscenes_infos_temporal_val_3keyframes.pkl", help="Path to info file")
    parser.add_argument("--view_order", type=list, default=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"], help="Order of camera views")
    parser.add_argument("--folder_count", type=int, default=7, help="Number of source folders")
    parser.add_argument("--gen_count", type=int, default=4, help="Number of generations")

    args = parser.parse_args()
    main(args)
