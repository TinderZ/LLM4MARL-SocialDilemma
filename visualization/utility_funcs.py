# visualization/utility_funcs.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def make_video_from_rgb_imgs(
    rgb_arrs: List[np.ndarray],
    vid_path: str,
    video_name: str = "trajectory",
    fps: int = 5,
    format: str = "mp4v",
    resize: Optional[Tuple[int, int]] = None
) -> str:
    """
    Creates an MP4 video from a list of RGB numpy arrays.

    Args:
        rgb_arrs: List of numpy arrays (frames), each of shape (H, W, 3).
        vid_path: Directory path to save the video.
        video_name: Name of the video file (without extension).
        fps: Frames per second for the video.
        format: FourCC code for the video codec (e.g., 'mp4v' for MP4).
        resize: Optional tuple (width, height) to resize frames.

    Returns:
        The full path to the saved video file.
    """
    print("Rendering video...")
    if not os.path.exists(vid_path):
        os.makedirs(vid_path, exist_ok=True)

    video_full_path = os.path.join(vid_path, f"{video_name}.mp4")

    if not rgb_arrs:
        print("Warning: No frames provided to create video.")
        return video_full_path

    # Determine frame size
    if resize:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, _ = frame.shape
        resize = (width, height) # Use original size if not specified

    # Ensure resize has width first, then height for cv2.VideoWriter
    output_size = (resize[0], resize[1])

    fourcc = cv2.VideoWriter_fourcc(*format)
    # Use float(fps) for compatibility
    video = cv2.VideoWriter(video_full_path, fourcc, float(fps), output_size)

    if not video.isOpened():
         print(f"Error: Could not open video writer for path: {video_full_path}")
         print(f"Codec: {format}, FPS: {float(fps)}, Size: {output_size}")
         return video_full_path # Return path even if writer failed

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if i % max(1, len(rgb_arrs) // 5) == 0: # Print progress ~5 times
            print(f"\t... {percent_done}% of frames rendered")

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # OpenCV expects BGR, but input is likely RGB from matplotlib/rendering
        # Convert RGB to BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize frame if necessary
        if img_bgr.shape[1::-1] != output_size: # shape is (h, w, c), size is (w, h)
             img_bgr = cv2.resize(img_bgr, output_size, interpolation=cv2.INTER_NEAREST)

        video.write(img_bgr)

    print(f"\t... 100% of frames rendered. Finalizing video at {video_full_path}")
    video.release()
    return video_full_path

def make_video_from_image_dir(
    vid_path: str,
    img_folder: str,
    video_name: str = "trajectory",
    fps: int = 5,
    pattern: str = "*.png",
    format: str = "mp4v",
    resize: Optional[Tuple[int, int]] = None
) -> Optional[str]:
    """
    Creates a video from a directory of image files (e.g., PNGs).

    Args:
        vid_path: Directory path to save the video.
        img_folder: Path to the directory containing image frames.
        video_name: Name of the video file (without extension).
        fps: Frames per second for the video.
        pattern: Glob pattern to find image files (e.g., "*.png", "frame*.jpg").
        format: FourCC code for the video codec.
        resize: Optional tuple (width, height) to resize frames.

    Returns:
        The full path to the saved video file, or None if no images found.
    """
    import glob

    image_files = sorted(glob.glob(os.path.join(img_folder, pattern)))

    if not image_files:
        print(f"Warning: No images found in {img_folder} matching pattern {pattern}.")
        return None

    print(f"Found {len(image_files)} images in {img_folder}.")

    # Read images and store as numpy arrays
    rgb_imgs = []
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            # Convert BGR (from cv2.imread) to RGB
            rgb_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            print(f"Warning: Could not read image file: {img_file}")

    if not rgb_imgs:
        print("Error: Failed to read any image files.")
        return None

    return make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name, fps, format, resize)

def get_all_subdirs(path):
    """Gets all subdirectories within a given path."""
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_all_files(path):
    """Gets all files (not directories) within a given path."""
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def save_img(rgb_arr, path, name):
    """Saves a numpy array as an image file."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)
    try:
        # Use matplotlib for consistency if needed, or directly use cv2
        # plt.imshow(rgb_arr, interpolation="nearest")
        # plt.savefig(full_path)
        # plt.close() # Close plot to free memory

        # Or using OpenCV (might be faster, ensure BGR conversion if needed)
        img_bgr = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(full_path, img_bgr)
        # print(f"Saved image to {full_path}")
    except Exception as e:
        print(f"Error saving image {full_path}: {e}")