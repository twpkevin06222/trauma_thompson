import torch
import argparse
import os
import json
import cv2
from utils import get_model
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import numpy as np
import time

# ignore pytorch warning
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the model checkpoint",
)
parser.add_argument(
    "--video_path",
    type=str,
    required=True,
    help="Path to video file or directory containing videos",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./inference_outputs",
    help="Directory to save output videos",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=124,
    help="Number of classes",
)
parser.add_argument(
    "--cls2idx_path",
    type=str,
    default="./outputs/cls2idx.json",
    help="Path to class to index mapping JSON file",
)
parser.add_argument(
    "--frame_length",
    type=int,
    default=16,
    help="Number of frames to extract from video",
)
parser.add_argument(
    "--frame_size",
    type=int,
    nargs=2,
    default=[256, 256],
    help="Frame size (height width)",
)
parser.add_argument(
    "--stride",
    type=int,
    default=8,
    help="Stride for overlapping windows (default: 8)",
)
args = parser.parse_args()

MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Load class mapping
with open(args.cls2idx_path, "r") as f:
    cls2idx = json.load(f)
idx2cls = {v: k for k, v in cls2idx.items()}


def temporal_moving_average(logits, window=3):
    """
    logits: torch.Tensor of shape [T, C]
        T = number of clips, C = num classes
    """
    smoothed = torch.zeros_like(logits)

    for i in range(logits.shape[0]):
        smoothed[i] = logits[i : i + window].mean(dim=0)

    return smoothed


def get_top_k_predictions(logits, k=5):
    """Get top-k predictions with probabilities"""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    return top_indices, top_probs


def write_predictions_on_video(
    video_path, predictions, probabilities, idx2cls, output_path, start_frame_list
):
    """Write predictions on video frames. Predictions are displayed starting from each clip's start frame and continue until the next clip begins."""
    print(f"Writing predictions on video: {output_path}")
    # Read original video with OpenCV to preserve quality
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a mapping of start_frame -> predictions for that clip
    # predictions and probabilities are tensors of shape [num_clips, 5]
    frame_to_predictions = {}
    for clip_idx, start_frame in enumerate(start_frame_list):
        frame_to_predictions[start_frame] = {
            "indices": predictions[clip_idx],
            "probs": probabilities[clip_idx],
        }

    # Keep track of current clip's predictions
    current_pred_indices = None
    current_pred_probs = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame is the start of a new clip
        if frame_count in frame_to_predictions:
            clip_preds = frame_to_predictions[frame_count]
            current_pred_indices = clip_preds["indices"]
            current_pred_probs = clip_preds["probs"]

        # Draw predictions on frame if we have current predictions
        if current_pred_indices is not None and current_pred_probs is not None:
            # Prepare text to display for top 5 predictions
            text_lines = []
            for i, (pred_idx, prob) in enumerate(
                zip(current_pred_indices, current_pred_probs)
            ):
                class_name = idx2cls[pred_idx.item()]
                prob_value = prob.item()
                text_lines.append(f"{i+1}. {class_name}: {prob_value:.3f}")

            # Draw predictions on frame
            y_offset = 30
            for i, text in enumerate(text_lines):
                # Draw background rectangle for better visibility
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (10, y_offset - text_height - 5),
                    (10 + text_width + 10, y_offset + 5),
                    (0, 0, 0),
                    -1,
                )
                # Draw text
                cv2.putText(
                    frame,
                    text,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 35

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def create_batches(init_frame, stop_frame, frame_length, overlap_frames):
    assert stop_frame > init_frame, "stop_frame must be greater than init_frame!"
    batch = (stop_frame - init_frame) // (frame_length - overlap_frames)
    start_frame_list = []
    end_frame_list = []
    for i in range(batch):
        start_frame = init_frame
        end_frame = start_frame + frame_length
        # before reaching the final batch
        if i == batch - 2:
            init_frame = stop_frame - frame_length
        else:
            init_frame += frame_length - overlap_frames
        start_frame_list.append(start_frame)
        end_frame_list.append(end_frame)
    return start_frame_list, end_frame_list


def inference_single_video(
    video_path,
    model,
    processor,
    idx2cls,
    output_dir,
    frame_length=16,
    frame_size=(256, 256),
    stride=8,
):
    """Run inference on a single video with overlapping windows"""
    print(f"Processing: {video_path}")
    # Load full video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_list, end_frame_list = create_batches(
        0, frame_count, frame_length, stride
    )
    start_frame_list, end_frame_list = create_batches(
        0, frame_count, frame_length, stride
    )
    # Process each window and collect logits
    model.eval()
    all_logits = []
    batch = len(start_frame_list)
    for i, (batch_start_frame, batch_end_frame) in enumerate(
        zip(start_frame_list, end_frame_list)
    ):
        print(
            f"Processing batch: {i+1}/{batch} clip from frame {batch_start_frame} to frame {batch_end_frame} .."
        )
        frames = []
        for frame_idx in range(batch_start_frame, batch_end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            frames.append(frame[np.newaxis, ...])
        frame_arr = np.concatenate(frames, axis=0)
        frame_arr = np.einsum("thwc->tchw", frame_arr)
        input_tensor = processor(frame_arr, return_tensors="pt").to(device)
        outputs = model(input_tensor.pixel_values_videos)
        logits = outputs.logits
        all_logits.append(logits)
    # Aggregate logits by averaging
    concat_logits = torch.concatenate(all_logits, dim=0)
    smoothed_logits = temporal_moving_average(concat_logits)
    # Get top-5 predictions from aggregated logits
    top_indices, top_probs = get_top_k_predictions(smoothed_logits, k=5)

    # Create output filename
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"annotated_{video_name}")

    # Write predictions on video
    write_predictions_on_video(
        video_path,
        top_indices,
        top_probs,
        idx2cls,
        output_path,
        start_frame_list,
    )

    print(f"Output saved to: {output_path}\n")
    return top_indices, top_probs


def main():
    # Load model
    print("Loading model...")
    model, processor = get_model(
        model_name=MODEL_NAME,
        num_classes=args.num_classes,
        verbosity=False,
        device=device,
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "head" in checkpoint:
            # If checkpoint contains only head weights
            model.classifier.load_state_dict(checkpoint["head"])
            print("Loaded classifier head from checkpoint.")
        elif "model" in checkpoint:
            # If checkpoint is wrapped in a dict with "model" key
            model.load_state_dict(checkpoint["model"])
            print("Loaded full model from checkpoint.")
        else:
            # Try loading as full model state dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded full model state dict from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load checkpoint as full model: {e}")
                print("Trying to load only classifier head...")
                if "classifier" in str(checkpoint.keys()):
                    # Try to extract classifier weights
                    for key in checkpoint.keys():
                        if "classifier" in key.lower() or "head" in key.lower():
                            model.classifier.load_state_dict(checkpoint[key])
                            print(f"Loaded classifier from key: {key}")
                            break
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}")
        print("Using untrained model weights.")

    model.eval()

    # Process video(s)
    if os.path.isfile(args.video_path):
        # Single video file
        inference_single_video(
            args.video_path,
            model,
            processor,
            idx2cls,
            args.output_dir,
            args.frame_length,
            tuple(args.frame_size),
            args.stride,
        )
    elif os.path.isdir(args.video_path):
        # Directory of videos
        video_files = [
            f
            for f in os.listdir(args.video_path)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]
        print(f"Found {len(video_files)} video(s) to process\n")

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(args.video_path, video_file)
            try:
                inference_single_video(
                    video_path,
                    model,
                    processor,
                    idx2cls,
                    args.output_dir,
                    args.frame_length,
                    tuple(args.frame_size),
                    args.stride,
                )
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}\n")
                continue
    else:
        print(f"Error: {args.video_path} is not a valid file or directory")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = int(total_time % 60)
    print(f"Inference time: {hours} hrs {minutes} mins {seconds} secs")
