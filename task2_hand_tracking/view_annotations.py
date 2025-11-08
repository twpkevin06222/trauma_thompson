"""
Helper script to view annotations for a video or batch process multiple videos

Usage (single video):
python view_annotations.py --annotations_dir /path/to/annotations --video_path /path/to/video --output_path /path/to/output --no-display

Usage (batch process all videos in a directory):
python view_annotations.py --annotations_dir /path/to/annotations --videos_dir /path/to/videos --output_path /path/to/output --no-display
"""

import os
import cv2
import json
import argparse
from glob import glob


def load_annotations_for_frame(ann_folder, frame_index, debug=False):
    # Try both naming conventions: frame_XXXXXX.json and XXXXXX.json
    ann_file = os.path.join(ann_folder, f"frame_{frame_index:06d}.json")
    if not os.path.exists(ann_file):
        # Fallback to format without "frame_" prefix
        ann_file = os.path.join(ann_folder, f"{frame_index:06d}.json")

    if not os.path.exists(ann_file):
        if debug and frame_index < 5:
            print(
                f"Debug: Annotation file not found for frame {frame_index}: {ann_file}"
            )
        return []

    with open(ann_file, "r") as f:
        data = json.load(f)

    bboxes = []
    annotations = data.get("annotations", [])
    if debug and frame_index < 5:
        print(f"Debug: Frame {frame_index} - Found {len(annotations)} annotation(s)")

    for ann in annotations:
        bbox = ann.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            bboxes.append(bbox)
            if debug and frame_index < 5:
                print(f"Debug: Frame {frame_index} - Added bbox: {bbox}")
        elif debug and frame_index < 5:
            print(f"Debug: Frame {frame_index} - Invalid bbox format: {bbox}")

    return bboxes


def play_video_with_annotations(
    video_path, annotations_folder, output_path=None, display=True, debug=False
):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False

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
                return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = load_annotations_for_frame(
            annotations_folder, frame_index, debug=debug
        )
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x2 = max(0, min(x + w, width - 1))
            y2 = max(0, min(y + h, height - 1))
            if x2 > x and y2 > y:  # Only draw if valid rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), color=(0, 255, 0), thickness=2)
                if debug and frame_index < 5:
                    print(
                        f"Debug: Frame {frame_index} - Drawing bbox: ({x}, {y}) to ({x2}, {y2})"
                    )

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

    return True


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
        help=("Path to a single video file to process"),
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        help=(
            "Directory containing videos to batch process (processes all .mp4 files)"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Output path for saving annotated videos. "
            "For single video: can be file path or directory. "
            "For batch processing: must be a directory where all annotated videos will be saved."
        ),
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Enable debug output to diagnose annotation loading issues"),
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.video_path and not args.videos_dir:
        parser.error("Either --video_path or --videos_dir must be provided")
    if args.video_path and args.videos_dir:
        parser.error("Cannot specify both --video_path and --videos_dir")
    if not args.annotations_dir:
        parser.error("--annotations_dir is required")

    # Auto-disable display when saving to file (unless explicitly enabled)
    if args.output_path and not args.display:
        display_enabled = False
    else:
        display_enabled = not args.no_display

    # Batch processing mode
    if args.videos_dir:
        if not os.path.isdir(args.videos_dir):
            parser.error(f"Videos directory does not exist: {args.videos_dir}")

        if not args.output_path:
            parser.error(
                "--output_path is required for batch processing (must be a directory)"
            )

        # Ensure output path is a directory
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
            print(f"Created output directory: {args.output_path}")

        # Find all video files
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob(os.path.join(args.videos_dir, ext)))
            video_files.extend(glob(os.path.join(args.videos_dir, ext.upper())))

        video_files = sorted(video_files)

        if not video_files:
            print(f"No video files found in {args.videos_dir}")
            exit(0)

        print(f"Found {len(video_files)} video(s) to process")

        # Process each video
        successful = 0
        failed = 0
        skipped = 0

        for i, video_path in enumerate(video_files, 1):
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            annotation_folder = os.path.join(args.annotations_dir, video_basename)
            output_filename = f"{video_basename}_annotated.mp4"
            output_video_path = os.path.join(args.output_path, output_filename)

            print(f"\n[{i}/{len(video_files)}] Processing: {video_basename}")

            if not os.path.exists(annotation_folder):
                print(f"  Warning: Annotation folder not found: {annotation_folder}")
                print(f"  Skipping {video_basename}")
                skipped += 1
                continue

            if args.debug:
                print(f"  Video path: {video_path}")
                print(f"  Annotation folder: {annotation_folder}")
                print(f"  Output path: {output_video_path}")

            success = play_video_with_annotations(
                video_path,
                annotation_folder,
                output_path=output_video_path,
                display=False,  # Always disable display for batch processing
                debug=args.debug,
            )

            if success:
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print("Batch processing complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total: {len(video_files)}")
        print(f"Output directory: {args.output_path}")
        print(f"{'='*60}")

    # Single video processing mode
    else:
        video_path = args.video_path
        annotation_folder = os.path.join(
            args.annotations_dir, os.path.splitext(os.path.basename(args.video_path))[0]
        )

        if args.debug:
            print(f"Debug: Video path: {video_path}")
            print(f"Debug: Annotation folder: {annotation_folder}")
            print(
                f"Debug: Annotation folder exists: {os.path.exists(annotation_folder)}"
            )
            if os.path.exists(annotation_folder):
                json_files = sorted(
                    [f for f in os.listdir(annotation_folder) if f.endswith(".json")]
                )
                print(f"Debug: Found {len(json_files)} JSON files in annotation folder")
                if json_files:
                    print(f"Debug: First few JSON files: {json_files[:5]}")
                    # Check first annotation file
                    first_file = os.path.join(annotation_folder, json_files[0])
                    with open(first_file, "r") as f:
                        first_data = json.load(f)
                        print(
                            f"Debug: First file structure - keys: {list(first_data.keys())}"
                        )
                        if "annotations" in first_data:
                            print(
                                f"Debug: First file - annotations count: {len(first_data['annotations'])}"
                            )
                            if len(first_data["annotations"]) > 0:
                                print(
                                    f"Debug: First annotation: {first_data['annotations'][0]}"
                                )

        play_video_with_annotations(
            video_path,
            annotation_folder,
            output_path=args.output_path,
            display=display_enabled,
            debug=args.debug,
        )
