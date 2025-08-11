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
        use_rgb=True,
        use_depth=False,
        val_ratio=0.0,  # Use all data for visualization
        max_train_episodes=None
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Available cameras: {dataset.camera_names}")
    
    # Get dataset info
    replay_buffer = dataset.replay_buffer
    n_episodes = replay_buffer.n_episodes
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
            for cam_name in dataset.camera_names:
                rgb_key = f'imgs/{cam_name}/rgb'
                if rgb_key not in episode_data:
                    continue
                    
                # Get RGB frame (H, W, C) and convert to uint8
                frame = episode_data[rgb_key][frame_idx]  # Shape: (H, W, 3)
                frame = (frame * 255).astype(np.uint8)
                
                # Initialize video writer if needed
                if cam_name not in video_writers:
                    if frame_shape is None:
                        frame_shape = frame.shape
                    
                    video_path = os.path.join(output_dir, f"{cam_name}_all_episodes.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers[cam_name] = cv2.VideoWriter(
                        video_path, fourcc, fps, (frame_shape[1], frame_shape[0])
                    )
                    print(f"Created video writer for {cam_name}: {video_path}")
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to video
                video_writers[cam_name].write(frame_bgr)
    
    # Close all video writers
    print("Finalizing videos...")
    for cam_name, writer in video_writers.items():
        writer.release()
        video_path = os.path.join(output_dir, f"{cam_name}_all_episodes.mp4")
        print(f"Video saved: {video_path}")
    
    # Print summary
    total_frames = sum(len(replay_buffer.get_episode(i)[dataset.action_key]) for i in range(n_episodes))
    video_duration = total_frames / fps
    
    print(f"\nSummary:")
    print(f"Total frames processed: {total_frames}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Videos created for cameras: {list(video_writers.keys())}")
    print(f"Output directory: {output_dir}")


def test_with_dataset_iteration(zarr_path):
    """
    Alternative test function that uses dataset iteration instead of direct replay buffer access.
    This method respects the dataset's sampling logic but may be slower.
    """
    # Configuration 
    output_dir = "dataset_videos_v2"
    fps = 30
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    dataset = DexmachinaImgDataset(
        zarr_path=zarr_path,
        horizon=3,  # Single frame sampling
        use_rgb=True,
        use_depth=False,
        val_ratio=0.0
    )

    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Available cameras: {dataset.camera_names}")
    
    # Initialize video writers
    video_writers = {}
    frame_shape = None
    
    print("Processing dataset samples...")
    
    # Iterate through all samples
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[idx]
        # Process each camera
        for cam_name in dataset.camera_names:
            rgb_key = f'imgs/{cam_name}/rgb'
            if rgb_key not in sample['obs']:
                continue
            
            # Get frame (already normalized to [0,1], shape: (1, 3, H, W))
            frame_tensor = sample['obs'][rgb_key][0]  # Remove time dimension
            
            # Convert to numpy and transpose to (H, W, C)
            frame = frame_tensor.permute(1, 2, 0).numpy()
            frame = (frame * 255).astype(np.uint8)
            
            # Initialize video writer if needed
            if cam_name not in video_writers:
                if frame_shape is None:
                    frame_shape = frame.shape
                
                video_path = os.path.join(output_dir, f"{cam_name}_all_episodes_v2.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writers[cam_name] = cv2.VideoWriter(
                    video_path, fourcc, fps, (frame_shape[1], frame_shape[0])
                )
                print(f"Created video writer for {cam_name}: {video_path}")
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video_writers[cam_name].write(frame_bgr)
    
    # Close all video writers
    print("Finalizing videos...")
    for cam_name, writer in video_writers.items():
        writer.release()
        video_path = os.path.join(output_dir, f"{cam_name}_all_episodes_v2.mp4")
        print(f"Video saved: {video_path}")
    
    print(f"\nVideos created for cameras: {list(video_writers.keys())}")
 
if __name__ == '__main__':
    zarr_path = '/home/mandi/devmachina/dexmachina/dataset/murp/no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16/replay_buffer.zarr'
    test_low_dim(zarr_path)
    
    # test_img_dataset()

    test_with_dataset_iteration(zarr_path)