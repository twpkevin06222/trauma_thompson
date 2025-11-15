import torch

from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_video
from torchvision.transforms import v2


def get_model(model_name, num_classes=124, verbosity=False, device="cuda"):
    processor = VJEPA2VideoProcessor.from_pretrained(model_name)
    model = VJEPA2ForVideoClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True
    ).to(device)
    # 1. Freeze the pretrained VJEPA2 backbone (encoder + predictor)
    for param in model.vjepa2.parameters():
        param.requires_grad = False
    for param in model.pooler.parameters():
        param.requires_grad = False
    # 2. Unfreeze only the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    if verbosity:
        total_params = sum(param.numel() for param in model.parameters())
        print()
        print("-" * 100)
        print(f"Total number of parameters: {total_params}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("Trainable:", name)
        print("-" * 100)
        print()
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes).to(device)
    # this is require to overwrite the config with the
    # proper number of labels
    model.config.num_labels = num_classes
    return model, processor


class VideoDataset(Dataset):
    def __init__(
        self,
        csv_pth,
        video_dir,
        frame_length=16,
        frame_size=(256, 256),
        transform=None,
        mode="train",
    ):
        self.csv_pth = csv_pth
        self.video_dir = video_dir
        self.transform = transform
        self.mode = mode
        # get train/val inputs
        self.df = pd.read_csv(csv_pth).query(f"split == '{mode}'")
        self.split_video_dir = os.path.join(video_dir, mode)
        self.eval_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.video_size = (
            frame_length,
            3,
            frame_size[0],
            frame_size[1],
        )  # (t, c, h, w)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_pth = os.path.join(
            self.split_video_dir, self.df.iloc[idx]["clip_id"] + ".mp4"
        )
        video, audio, video_meta = read_video(video_pth, pts_unit="sec")
        video = torch.einsum("thwc->tchw", video)  # convert to (t, c, h, w)
        # video has different time length during extraction process
        try:
            assert video.shape == self.video_size, "Video shape mismatch"
        except:
            pad_t = self.video_size[0] - video.shape[0]
            if pad_t <= 5:
                # if the video is too short, we will pad it with the last (t - t_n) frames
                print(
                    f"{video_pth} has shape {video.shape} but expected {self.video_size}"
                )
                print("Proceed with padding ...")
                # pad the video the final (t - t_n) frames
                pad = video[video.shape[0] - pad_t : video.shape[0]]
                video = torch.cat((video, pad), dim=0)
            else:
                print(
                    f"Video is too short, skipping ..."
                    f"{video_pth} has shape {video.shape} but expected {self.video_size}"
                )
                print("Resampling with new index ...")
                new_idx = np.random.randint(0, len(self.df))
                return self.__getitem__(new_idx)
        if (self.transform is not None) and (self.mode == "train"):
            video = self.transform(video)
        else:
            video = self.eval_transform(video)
        return video, self.df.iloc[idx]["action_idx"]


def get_video_transform(prob=0.25):
    video_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=prob),  # Randomly flip horizontally
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
                p=prob,
            ),
            v2.RandomApply([v2.RandomRotation(degrees=10)], p=prob),
            v2.RandomApply(
                [
                    v2.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                    )
                ],
                p=prob,
            ),
            v2.RandomPerspective(distortion_scale=0.2, p=prob),
            v2.RandomGrayscale(p=0.2),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=prob),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))], p=prob
            ),
            v2.RandomAutocontrast(p=prob),
            v2.RandomEqualize(p=prob),
            v2.RandomErasing(p=prob),
            v2.RandomInvert(p=prob),
            v2.RandomPosterize(bits=4, p=prob),
            v2.RandomSolarize(threshold=128, p=prob),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return video_transform
