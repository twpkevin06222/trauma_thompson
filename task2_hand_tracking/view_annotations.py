"""
Helper script to view annotations for a video

Usage:
python view_annotations.py --annotations_dir /path/to/annotations --video_path /path/to/video --output_path /path/to/output --no-display
python view_annotations.py --annotations_dir /path/to/annotations --video_path /path/to/video --output_path /path/to/output --display
python view_annotations.py --annotations_dir /path/to/annotations --video_path /path/to/video --output_path /path/to/output --no-display --display
"""

import os
import cv2
import json
import argparse


def load_annotations_for_frame(ann_folder, frame_index):
    # Try both naming conventions: frame_XXXXXX.json and XXXXXX.json
    ann_file = os.path.join(ann_folder, f"frame_{frame_index:06d}.json")
    if not os.path.exists(ann_file):
        # Fallback to format without "frame_" prefix
        ann_file = os.path.join(ann_folder, f"{frame_index:06d}.json")

    if not os.path.exists(ann_file):
        return []

    with open(ann_file, "r") as f:
        data = json.load(f)

    bboxes = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            bboxes.append(bbox)
    return bboxes


def play_video_with_annotations(
    video_path, annotations_folder, output_path=None, display=True
):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        # Handle output path: if it's a directory or has no extension, generate filename
        if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
            # Generate filename from input video name
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{video_basename}_annotated.mp4"
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, output_filename)
            else:
                # No extension provided, add .mp4
                output_path = (
                    f"{output_path}.mp4"
                    if not output_path.endswith(".")
                    else f"{output_path}mp4"
                )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Try different codecs for better compatibility
        fourcc_options = [
            cv2.VideoWriter_fourcc(*"mp4v"),
            cv2.VideoWriter_fourcc(*"XVID"),
            cv2.VideoWriter_fourcc(*"avc1"),  # H.264
        ]

        writer = None
        for fourcc in fourcc_options:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Saving annotated video to: {output_path}")
                break

        if writer is None or not writer.isOpened():
            print(
                f"Warning: Failed to open video writer for {output_path}. Attempting without codec."
            )
            writer = cv2.VideoWriter(output_path, -1, fps, (width, height))
            if not writer.isOpened():
                print(
                    f"Error: Could not create video writer. Check permissions and path: {output_path}"
                )
                return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = load_annotations_for_frame(annotations_folder, frame_index)
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        # Write frame to output video if writer is available
        if writer is not None:
            writer.write(frame)

        # Display frame if display is enabled
        if display:
            try:
                cv2.imshow("Video with Hand Detection", frame)
                key = cv2.waitKey(30)  # Press 'q' to quit
                if key == ord("q"):
                    break
            except cv2.error as e:
                print(
                    f"Display error: {e}. Disabling display and continuing to save video."
                )
                display = False

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video saved successfully: {output_path}")
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper script to view annotations for a video"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        help=("Directory containing the annotations"),
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help=("Name of the video to view"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=("Output path for saving the annotated video (optional)"),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help=("Disable video display (useful when only saving to file)"),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Explicitly enable video display (default: auto-detect based on output_path)"
        ),
    )
    args = parser.parse_args()

    video_path = args.video_path
    annotation_folder = os.path.join(
        args.annotations_dir, os.path.splitext(os.path.basename(args.video_path))[0]
    )

    # Auto-disable display when saving to file (unless explicitly enabled)
    if args.output_path and not args.display:
        display_enabled = False
    else:
        display_enabled = not args.no_display

    play_video_with_annotations(
        video_path,
        annotation_folder,
        output_path=args.output_path,
        display=display_enabled,
    )
