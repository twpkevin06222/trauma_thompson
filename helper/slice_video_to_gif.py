import cv2
import imageio
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument(
    "--duration_seconds", type=int, default=10, help="Gif duration in seconds"
)
parser.add_argument("--fps", type=int, default=10, help="Gif desired frames per second")
parser.add_argument(
    "--resize",
    type=tuple,
    default=(640, 480),
    help="Resize frames to the specified dimensions (width, height)",
)
args = parser.parse_args()


def video_to_gif(
    video_path, output_path=None, duration_seconds=10, fps=10, resize=None
):
    """
    Convert a video to GIF, trimming it to the specified duration.

    Args:
        video_path: Path to input video file
        output_path: Path to output GIF file (if None, uses video_path with .gif extension)
        duration_seconds: Maximum duration of the GIF in seconds (default: 10)
        fps: Frames per second for the GIF (default: 10)
        resize: Resize frames. Can be:
            - None: no resizing (default)
            - (width, height): specific dimensions
            - (width, None) or (None, height): maintain aspect ratio
            - float: scale factor (e.g., 0.5 for 50% size)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / original_fps if original_fps > 0 else 0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate resize dimensions
    resize_width, resize_height = None, None
    if resize is not None:
        if isinstance(resize, (int, float)):
            # Scale factor
            resize_width = int(original_width * resize)
            resize_height = int(original_height * resize)
        elif isinstance(resize, (tuple, list)) and len(resize) == 2:
            width, height = resize
            if width is None and height is None:
                resize_width, resize_height = None, None
            elif width is None:
                # Calculate width to maintain aspect ratio
                aspect_ratio = original_width / original_height
                resize_width = int(height * aspect_ratio)
                resize_height = int(height)
            elif height is None:
                # Calculate height to maintain aspect ratio
                aspect_ratio = original_height / original_width
                resize_width = int(width)
                resize_height = int(width * aspect_ratio)
            else:
                resize_width, resize_height = int(width), int(height)

    # Calculate how many frames to extract
    max_frames = (
        int(duration_seconds * original_fps)
        if original_fps > 0
        else int(duration_seconds * 30)
    )
    frames_to_extract = min(max_frames, total_frames)

    # Calculate frame interval to maintain desired fps
    frame_interval = max(1, int(original_fps / fps)) if original_fps > 0 else 1

    print("Video info:")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total duration: {total_duration:.2f} seconds")
    print(f"  Original resolution: {original_width}x{original_height}")
    if resize_width and resize_height:
        print(f"  Resized resolution: {resize_width}x{resize_height}")
    print(f"  Extracting {frames_to_extract} frames (first {duration_seconds} seconds)")
    print(f"  GIF FPS: {fps}")

    # Extract frames
    frames = []
    frame_count = 0
    extracted_count = 0

    while frame_count < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at intervals to match desired fps
        if frame_count % frame_interval == 0:
            # Resize frame if needed
            if resize_width and resize_height:
                frame = cv2.resize(
                    frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA
                )
            # Convert BGR to RGB (OpenCV uses BGR, imageio uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1

        frame_count += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError("Error: No frames extracted from video")

    print(f"Extracted {len(frames)} frames")

    # Determine output path
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.with_suffix(".gif"))

    # Create GIF
    print(f"Creating GIF: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

    print(f"GIF created successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    video_pth = args.video_path
    output_path = args.output_path
    duration_seconds = args.duration_seconds
    fps = args.fps
    resize = tuple(args.resize)
    try:
        output_gif = video_to_gif(
            video_pth, duration_seconds=duration_seconds, fps=fps, resize=resize
        )
        print(f"\nSuccess! Output saved to: {output_gif}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
