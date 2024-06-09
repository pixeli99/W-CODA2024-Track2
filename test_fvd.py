import os
import json
import torch
from PIL import Image
from torchvision import transforms
from calculate_fvd import calculate_fvd
import mmcv
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FVD')
    parser.add_argument('--base_path', type=str, default="/MagicDrive/magicdrive-t-log/test", help='Base path')
    parser.add_argument('--data_path', type=str, default="/BEVFormer_segmentation_detection/data/nuscenes/nuscenes_infos_temporal_val_12hz.pkl", help='Data path')
    parser.add_argument('--output_path1', type=str, default="/videos1.pt", help='Output path for videos1')
    parser.add_argument('--output_path2', type=str, default="/videos2.pt", help='Output path for videos2')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for computation')
    parser.add_argument('--use_cache', type=bool, default=True, help='Whether to use cached Tensor')
    args = parser.parse_args()
    return args

def main(args):
    # Define paths and parameters
    folders = [f"SDv1.5mv-rawbox-t_2024-06-06_10-22_{i}" for i in range(1, 8)]
    view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
    device = torch.device(args.device)  # Use GPU for acceleration

    # Load data
    data = mmcv.load(args.data_path)

    # Define parameters
    NUMBER_OF_SCENES = 150
    VIDEO_LENGTH = 16  # Each video has 16 frames
    CHANNEL = 3
    NUMBER_OF_GENERATIONS = 4
    NUMBER_OF_VIEWS = len(view_order)

    # Create an empty tensor to store real video data
    if args.use_cache:
        videos2 = torch.load(args.output_path2)
        print("Original video data loaded from cache")
    else:
        videos2 = torch.zeros(NUMBER_OF_SCENES * NUMBER_OF_VIEWS, VIDEO_LENGTH, CHANNEL, 224, 400, requires_grad=False)

        # Function: Read and preprocess a frame image
        def read_and_preprocess_image(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (400, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1)  # Convert to (CHANNEL, HEIGHT, WIDTH)
            image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
            return image

        def load_scene_data(scene_idx):
            results = []
            for view_idx, view in enumerate(view_order):
                scene_images = []
                for frame_idx in range(VIDEO_LENGTH):
                    info_idx = scene_idx * 16 + frame_idx
                    image_path = data['infos'][info_idx]['cams'][view]['data_path']
                    image_path = os.path.join('/', image_path)
                    image = read_and_preprocess_image(image_path)
                    scene_images.append(image)
                results.append(scene_images)
            return results

        # Use a thread pool to process data in parallel
        print("Loading original video data...")
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(load_scene_data, scene_idx) for scene_idx in range(NUMBER_OF_SCENES)]
            idx = 0
            for future in tqdm(as_completed(futures), total=NUMBER_OF_SCENES, desc="Loading scenes"):
                for view_images in future.result():
                    videos2[idx] = torch.stack(view_images)
                    idx += 1

        print("Original video data has been organized into a tensor")
        torch.save(videos2, args.output_path2)
        print("Original video data saved to:", args.output_path2)

    def process_image(gen_path, view_idx, transform):
        gen_image = Image.open(gen_path)
        width, height = gen_image.size
        assert width == 2400 and height == 224, "Image size does not meet expectations"

        left = view_idx * 400
        right = (view_idx + 1) * 400

        gen_view = gen_image.crop((left, 0, right, height))
        gen_tensor = transform(gen_view)
        return gen_tensor

    def process_scene(folder, scene, view_idx, generation, base_path, transform, videos1, VIDEO_LENGTH, index):
        for frame_idx in range(VIDEO_LENGTH):
            gen_file = f"{scene+1}_{frame_idx}_gen{generation}_{scene*16+frame_idx}.png"
            gen_path = os.path.join(base_path, folder, "frames", gen_file)

            if not os.path.isfile(gen_path):
                continue

            gen_tensor = process_image(gen_path, view_idx, transform)
            videos1[index, frame_idx] = gen_tensor

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

    if args.use_cache:
        videos1 = torch.load(args.output_path1)
        print("Generated video data loaded from cache")
    else:
        index = 0
        videos1 = torch.zeros(NUMBER_OF_SCENES * NUMBER_OF_GENERATIONS * NUMBER_OF_VIEWS, VIDEO_LENGTH, CHANNEL, 224, 400, requires_grad=False)
        transform = transforms.Compose([transforms.ToTensor()])  # your transform

        index = 0
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder in folders:
                folder_path = os.path.join(args.base_path, folder)
                number_of_scenes = get_number_of_scenes(folder_path)
                print(f"Processing folder: {folder}, Number of scenes: {number_of_scenes}")
                for scene in range(number_of_scenes):
                    for generation in range(NUMBER_OF_GENERATIONS):
                        for view_idx, view in enumerate(view_order):
                            futures.append(executor.submit(process_scene, folder, scene, view_idx, generation, args.base_path, transform, videos1, VIDEO_LENGTH, index))
                            index += 1

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
                pass

        torch.save(videos1, args.output_path1)
        print("Generated video data saved to:", args.output_path1)

    # Calculate FVD
    print("Calculating FVD...")
    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')

    # Print the result
    print(json.dumps(result, indent=4))

if __name__ == '__main__':
    args = parse_args()
    main(args)
