import os
import sys
import fire
from tqdm import tqdm
from glob import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as TF
from moviepy.editor import VideoFileClip


class VidPathDataset(torch.utils.data.Dataset):
    def __init__(self, listoflist, transform) -> None:
        self.data_list = listoflist
        self.transform = transform

    def __getitem__(self, idx):
        li = self.data_list[idx]
        clip = VideoFileClip(li)
        frames = clip.iter_frames()
        imgs = []
        for idx, frame in enumerate(frames):
            img = Image.fromarray(frame).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs

    def __len__(self):
        return len(self.data_list)


def top_center_crop(img, target_size):
    fH, fW = target_size  # see mmdet3d, this is inversed
    newW, newH = img.size
    crop_h = newH - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    img = img.crop(crop)
    return img


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


# get fid score
IN_SIZE = (900, 1600)
TARGET_SIZE = (224, 400)
resize_ratio = 0.25


def main(
    vid_root="magicdrive-t-log/submission/video",
    out_dir="magicdrive-t-log/evaluation/fvd",
    method="videogpt",
    load_from="../pretrained/fvd/videogpt", 
):
    clip_list = glob(os.path.join(vid_root, "*/*.mp4"))
    print(f"I got {len(clip_list)} videos.")

    _size = (int(IN_SIZE[0] * resize_ratio), int(IN_SIZE[1] * resize_ratio))
    transforms = TF.Compose([
        TF.Resize(_size, interpolation=TF.InterpolationMode.BICUBIC),
        lambda x: top_center_crop(x, target_size=TARGET_SIZE),
        TF.ToTensor(),
    ])
    dataset = VidPathDataset(clip_list, transforms)

    dims = 400
    batch_size = 50
    num_workers = 4
    device = "cuda"
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from fvd.videogpt.fvd import frechet_distance
    i3d = load_i3d_pretrained(device=device, load_from=load_from)

    pred_arr = np.empty((len(dataset), dims))
    start_idx = 0
    first_batch = True
    for batch in tqdm(dataloader, ncols=80):
        if first_batch:
            print(f"size of batch: {batch.shape}")
            first_batch = False

        with torch.no_grad():
            batch = trans(batch)
            feats1 = get_fvd_feats(batch, i3d=i3d, device=device)

        pred = feats1.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "fvd_feats_gen.npy"), pred_arr)


if __name__ == "__main__":
    fire.Fire(main)
