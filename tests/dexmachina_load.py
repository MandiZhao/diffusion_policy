import os
import cv2
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusion_policy.dataset.dexmachina_datasets import DexmachinaLowDimDataset, DexmachinaImgDataset
import matplotlib.pyplot as plt  # For colormap

def apply_jet_colormap(depth_img_3ch: np.ndarray) -> np.ndarray:
    """
    Convert a 3-channel repeated depth image to a colormap visualization.

    Args:
        depth_img_3ch: numpy array, shape (H, W, 3), float32 or uint8 assumed in [0,1] or [0,255]

    Returns:
        BGR uint8 image (H, W, 3) colored with jet colormap.
    """
    # Convert to single channel by taking the first channel (since repeated)
    depth_single = depth_img_3ch[:, :, 0]

    # Normalize depth to [0, 1] if needed
    if depth_single.dtype == np.uint8:
        depth_norm = depth_single.astype(np.float32) / 255.0
    else:
        # Assume float input - normalize to 0..1 by min/max
        min_val = np.min(depth_single)
        max_val = np.max(depth_single)
        if max_val - min_val > 1e-5:
            depth_norm = (depth_single - min_val) / (max_val - min_val)
        else:
            depth_norm = np.zeros_like(depth_single)

    # Apply jet colormap from matplotlib (returns RGBA)
    cmap = plt.get_cmap('jet')
    depth_colored = cmap(depth_norm)[:, :, :3]  # Drop alpha channel

    # Convert from RGB float [0,1] to BGR uint8 [0,255]
    depth_bgr = (depth_colored[..., ::-1] * 255).astype(np.uint8)

    return depth_bgr

def test_low_dim(zarr_path):
    lowdim_dataset = DexmachinaLowDimDataset(
        zarr_path=zarr_path,
        horizon=8,
        pad_before=1,
        pad_after=0
    )
    sample = lowdim_dataset[0]
    breakpoint()
    print(f"LowDim Dataset length: {len(lowdim_dataset)}")

def test_img_dataset(zarr_path):
    """
    Test function to load DexmachinaImgDataset and create videos for each camera
    showing all episodes concatenated together.
    """
    # Configuration
    output_dir = "dataset_videos"
    fps = 30

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    # Load dataset with validation split to get all data
    dataset = DexmachinaImgDataset(
        zarr_path=zarr_path,
        horizon=1,  # Single frame sampling
        pad_before=0,
        pad_after=0,
        max_train_episodes=2,
        state_keys=['robot/dof_pos'],
        # camera_keys=['imgs/front_256/depth', 'imgs/front_256/rgb'],
        camera_keys=['imgs/front_320/depth', 'imgs/front_320/rgb'],

    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Available cameras: {dataset.camera_keys}")

    # Get dataset info
    replay_buffer = dataset.replay_buffer
    n_episodes = 3
    print(f"Number of episodes: {n_episodes}")

    # Initialize video writers for each camera
    video_writers = {}
    frame_shape = None

    print("Processing episodes and creating videos...")

    # Process each episode
    for ep_idx in tqdm(range(n_episodes), desc="Processing episodes"):
        episode_data = replay_buffer.get_episode(ep_idx)
        episode_length = len(episode_data[dataset.action_key])

        print(f"Episode {ep_idx}: {episode_length} frames")

        # Process each frame in the episode
        for frame_idx in range(episode_length):
            # Get frame data for each camera
            for key in dataset.camera_keys:
                is_depth = 'depth' in key
                is_rgb = 'rgb' in key
                if not (is_depth or is_rgb):
                    continue
                cam_name = key

                # Get RGB frame (C, H, W) and convert to uint8 (H, W, C)
                frame = episode_data[key][frame_idx] # np.array
                frame = frame.transpose(1, 2, 0)

                if is_rgb:
                    frame = (frame * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = apply_jet_colormap(frame)

                # Initialize video writer if needed
                if cam_name not in video_writers:
                    if frame_shape is None:
                        frame_shape = frame.shape
                    name = cam_name.replace('/', '_')
                    video_path = os.path.join(output_dir, f"{name}_all_episodes.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers[cam_name] = cv2.VideoWriter(
                        video_path, fourcc, fps, (frame_shape[1], frame_shape[0])
                    )
                    print(f"Created video writer for {cam_name}: {video_path}")
                # Write frame to video
                video_writers[cam_name].write(frame_bgr)

    # Close all video writers
    print("Finalizing videos...")
    for cam_name, writer in video_writers.items():
        writer.release()
        name = cam_name.replace('/', '_')
        video_path = os.path.join(output_dir, f"{name}_all_episodes.mp4")
        print(f"Video saved: {video_path}")

    # Print summary
    total_frames = sum(len(replay_buffer.get_episode(i)[dataset.action_key]) for i in range(n_episodes))
    video_duration = total_frames / fps

    print(f"\nSummary:")
    print(f"Total frames processed: {total_frames}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Videos created for cameras: {list(video_writers.keys())}")
    print(f"Output directory: {output_dir}")

def test_img_batch(zarr_path, batch_size=8):
    # Load dataset
    dataset = DexmachinaImgDataset(
        zarr_path=zarr_path,
        horizon=16,
        max_train_episodes=1,
        camera_keys=['imgs/front_320/depth', 'imgs/front_320/rgb'],
        state_keys=['robot/dof_pos']
    )
    output_dir = "dataset_videos"
    # Create depth augmentor (example params)
    # depth_augmentor = DepthImageAugmentor(noise_std=0.02, dropout_prob=0.3)

    # Sample a batch of indices randomly
    indices = np.random.choice(len(dataset), batch_size, replace=False)

    # Prepare video writers for each camera (B, T=1, C, H, W)
    video_writers = {}
    fps = 10

    for idx in indices:
        sample = dataset[idx]  # dict: {'obs': {cam_key: tensor}, 'action': tensor}

        for cam_key in dataset.camera_keys:
            is_depth = 'depth' in cam_key
            is_rgb = 'rgb' in cam_key
            if not (is_depth or is_rgb):
                continue

            frame = sample['obs'][cam_key]  # tensor shape [T=1, C, H, W]
            frame = frame.squeeze(0).cpu().numpy()  # (C, H, W)

            # Apply depth augmentor if depth
            if is_depth:
                frame_tensor = torch.from_numpy(frame).unsqueeze(0)  # add batch dim
                frame = frame_tensor.squeeze(0).cpu().numpy()

            # Convert to HWC
            frame = np.transpose(frame, (1, 2, 0))

            if is_rgb:
                img = (frame * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:  # depth: visualize with jet colormap
                img_bgr = apply_jet_colormap(frame)

            # Init video writer per cam if needed
            if cam_key not in video_writers:
                h, w = img_bgr.shape[:2]
                video_path = f"{output_dir}/{cam_key.replace('/', '_')}_batch.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writers[cam_key] = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

            video_writers[cam_key].write(img_bgr)

    for writer in video_writers.values():
        writer.release()
    print("Batch videos saved for cameras:", list(video_writers.keys()))


if __name__ == '__main__':
    # zarr_path = '/home/mandi/devmachina/dexmachina/dataset/murp/no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16/replay_buffer.zarr'
    # zarr_path = '/checkpoint/siro/mandizhao/devmachina/dexmachina/dataset/murp/ep500/no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16/replay_buffer.zarr'
    zarr_path = '/checkpoint/siro/mandizhao/devmachina/dexmachina/dataset/murp/cam320/no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16/replay_buffer.zarr'

    # test_low_dim(zarr_path)

    # test_img_dataset(zarr_path)

    test_img_batch(zarr_path)
