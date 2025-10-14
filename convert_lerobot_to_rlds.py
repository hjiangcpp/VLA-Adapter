#!/usr/bin/env python

"""
Script to convert a LeRobot dataset to RLDS format and save it as a TensorFlow dataset.
"""

import argparse
import os
import sys
import numpy as np
import torch
import traceback  # Add this to get detailed error information
from tqdm import tqdm  # Add this import for the progress bar
import h5py
import cv2  # For video processing

# Set environment variables to force offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Add the path to the LeRobot library
# Adjust this path to point to the directory containing the lerobot package
sys.path.append('/home/hhjiang/Documents/VLA-Adapter')  # Update this path to where your lerobot code is located

def extract_frames_from_video(video_path, target_size=(224, 224)):
    """
    Extract all frames from a video file and resize them.
    
    Args:
        video_path: Path to the video file
        target_size: Tuple of (width, height) to resize frames to
        
    Returns:
        List of frames as numpy arrays in HWC format (uint8)
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {frame_count} frames from {video_path.name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, target_size)
        
        frames.append(frame_resized)
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames

# Function to load your dataset
def load_dataset_and_save_to_disk(repo_id, root=None, local_files_only=False, output_dir=None):
    """
    Load a dataset by repo_id using direct file access to avoid network calls.
    
    Args:
        repo_id: Repository ID of the dataset
        root: Root directory for the dataset stored locally
        local_files_only: Use local files only
        
    Returns:
        A dataset object in the format expected by DatasetToRLDSConverter
    """
    try:
        import pandas as pd
        import json
        from pathlib import Path
        import cv2  # For video processing
        
        # Construct the dataset path
        dataset_path = Path(root) / repo_id if root else Path(repo_id)
        
        print(f"Loading dataset from: {dataset_path}")
        
        # Check if dataset exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Load metadata
        info_path = dataset_path / "meta" / "info.json"
        episodes_path = dataset_path / "meta" / "episodes.jsonl"
        
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"Dataset info: {info}")
        
        # Load episodes metadata
        episodes_meta = []
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                for line in f:
                    episodes_meta.append(json.loads(line.strip()))
        
        # Find all parquet files
        data_dir = dataset_path / "data"
        parquet_files = list(data_dir.glob("**/*.parquet"))
        parquet_files.sort()
        
        print(f"Found {len(parquet_files)} parquet files")
        
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in dataset")
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, ""), exist_ok=True)
        
        # Process each parquet file (each represents an episode)
        for episode_idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing episodes")):
            print(f"Processing episode {episode_idx}: {parquet_file.name}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_file)
            
            print(f"Episode {episode_idx} has {len(df)} frames")
            print(f"Columns: {list(df.columns)}")
            
            # Load video data for this episode
            video_data = {}
            videos_dir = dataset_path / "videos" / "chunk-000"
            
            # Define camera names based on the metadata
            camera_names = ["observation.images.top", "observation.images.hand", "observation.images.front"]
            
            for camera_name in camera_names:
                video_path = videos_dir / camera_name / f"episode_{episode_idx:06d}.mp4"
                if video_path.exists():
                    print(f"Loading video: {camera_name}")
                    frames = extract_frames_from_video(video_path)
                    if frames:
                        video_data[camera_name.replace('.', '_')] = frames
                else:
                    print(f"Warning: Video not found: {video_path}")
            
            # Convert to HDF5 format
            with h5py.File(os.path.join(output_dir, f"episode_{episode_idx:06d}.h5"), 'w') as f:
                # Store metadata
                f.attrs['file_path'] = f"episode_{episode_idx:06d}"
                f.attrs['original_file'] = str(parquet_file)
                
                # Create a group for frames data
                frames_group = f.create_group('frames')
                
                num_frames = len(df)
                
                # Process each column in the dataframe
                for col in df.columns:
                    print(f"Processing column: {col}")
                    
                    # Get the first non-null value to determine data type and shape
                    first_valid_idx = df[col].first_valid_index()
                    if first_valid_idx is None:
                        print(f"Skipping column {col} - all values are null")
                        continue
                    
                    first_value = df[col].iloc[first_valid_idx]
                    
                    if isinstance(first_value, np.ndarray):
                        # Handle numpy arrays (states, actions)
                        print(f"Column {col} - numpy array shape: {first_value.shape}, dtype: {first_value.dtype}")
                        
                        sample_shape = first_value.shape
                        dtype = first_value.dtype
                        
                        # Create dataset
                        dataset = frames_group.create_dataset(
                            col.replace('.', '_'),
                            shape=(num_frames, *sample_shape),
                            dtype=dtype
                        )
                        
                        # Fill dataset
                        for idx, value in enumerate(df[col]):
                            if value is not None:
                                dataset[idx] = value
                            else:
                                dataset[idx] = np.zeros(sample_shape, dtype=dtype)
                    
                    elif isinstance(first_value, (int, float, np.integer, np.floating)):
                        # Handle scalar values
                        print(f"Column {col} - scalar dtype: {type(first_value)}")
                        dtype = np.float32 if isinstance(first_value, (float, np.floating)) else np.int32
                        dataset = frames_group.create_dataset(
                            col.replace('.', '_'),
                            shape=(num_frames,),
                            dtype=dtype
                        )
                        
                        for idx, value in enumerate(df[col]):
                            dataset[idx] = value if value is not None else 0
                    
                    elif isinstance(first_value, str):
                        # Handle string values
                        print(f"Column {col} - string")
                        dataset = frames_group.create_dataset(
                            col.replace('.', '_'),
                            shape=(num_frames,),
                            dtype=h5py.special_dtype(vlen=str)
                        )
                        
                        for idx, value in enumerate(df[col]):
                            dataset[idx] = value if value is not None else ""
                    
                    else:
                        print(f"Skipping column {col} - unsupported type: {type(first_value)}")
                
                # Add video data (camera images)
                for camera_key, frames in video_data.items():
                    if frames:
                        print(f"Adding camera data: {camera_key}")
                        # Verify we have the right number of frames
                        if len(frames) != num_frames:
                            print(f"Warning: Video {camera_key} has {len(frames)} frames, but parquet has {num_frames}. Adjusting...")
                            # Pad or truncate frames to match parquet data
                            if len(frames) < num_frames:
                                # Pad with last frame
                                while len(frames) < num_frames:
                                    frames.append(frames[-1])
                            else:
                                # Truncate
                                frames = frames[:num_frames]
                        
                        # Create dataset for camera images
                        frame_shape = frames[0].shape  # Should be (H, W, C)
                        camera_dataset = frames_group.create_dataset(
                            camera_key,
                            shape=(num_frames, *frame_shape),
                            dtype=np.uint8
                        )
                        
                        # Fill with video frames
                        for idx, frame in enumerate(frames):
                            camera_dataset[idx] = frame
                
                # Add language instruction (empty for now)
                lang_dataset = frames_group.create_dataset(
                    'language_instruction',
                    shape=(num_frames,),
                    dtype=h5py.special_dtype(vlen=str)
                )
                for idx in range(num_frames):
                    lang_dataset[idx] = ""
                
                # Add dataset metadata
                if info:
                    for key, value in info.items():
                        if isinstance(value, (str, int, float)):
                            f.attrs[f'info_{key}'] = value
        
        print(f"Successfully converted {len(parquet_files)} episodes to HDF5 format")
        print(f"Output saved to: {output_dir}")
        
        # Generate RLDS metadata files
        generate_rlds_metadata(output_dir, parquet_files, info, repo_id)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(traceback.format_exc())
        
        # Create a minimal dataset with one empty episode for testing
        print("Creating a dummy dataset for testing...")
        os.makedirs(os.path.join(output_dir, ""), exist_ok=True)
        
        with h5py.File(os.path.join(output_dir, "episode_000000.h5"), 'w') as f:
            f.attrs['file_path'] = "episode_000000"
            frames_group = f.create_group('frames')
            
            # Create dummy data
            frames_group.create_dataset('action', data=np.zeros((1, 7), dtype=np.float32))
            frames_group.create_dataset('state', data=np.zeros((1, 10), dtype=np.float32))
            frames_group.create_dataset('observation_image', data=np.zeros((1, 224, 224, 3), dtype=np.uint8))
            
            # String dataset
            string_dt = h5py.special_dtype(vlen=str)
            lang_dataset = frames_group.create_dataset('language_instruction', (1,), dtype=string_dt)
            lang_dataset[0] = "test instruction"


def generate_rlds_metadata(output_dir, parquet_files, dataset_info, repo_id):
    """
    Generate dataset_info.json and features.json files for RLDS format.
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Analyze the first parquet file to understand the data structure
    first_parquet = parquet_files[0]
    df = pd.read_parquet(first_parquet)
    
    print(f"Analyzing data structure from: {first_parquet.name}")
    print(f"Columns: {list(df.columns)}")
    
    # Count total episodes and calculate sizes
    total_episodes = len(parquet_files)
    total_frames = sum(len(pd.read_parquet(pf)) for pf in parquet_files)
    
    # Estimate file size (rough calculation)
    estimated_bytes = str(total_frames * 1000)  # Rough estimate
    
    # Generate dataset_info.json
    dataset_info_json = {
        "citation": f"// Dataset: {repo_id}",
        "description": f"Converted LeRobot dataset: {repo_id}\\nThis dataset contains robot demonstration data with actions, observations, and metadata.",
        "fileFormat": "tfrecord", 
        "moduleName": f"{repo_id.replace('-', '_')}.{repo_id.replace('-', '_')}_dataset_builder",
        "name": repo_id.replace('-', '_'),
        "releaseNotes": {
            "1.0.0": "Initial release."
        },
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": estimated_bytes,
                "shardLengths": [str(total_episodes)]
            }
        ],
        "version": "1.0.0"
    }
    
    # Generate features.json based on the actual data structure
    features_dict = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "feature": {
                        "observation": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {}
                            }
                        },
                        "action": {
                            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                            "tensor": {
                                "shape": {"dimensions": []},
                                "dtype": "float32",
                                "encoding": "none"
                            },
                            "description": "Robot action."
                        },
                        "language_instruction": {
                            "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                            "text": {},
                            "description": "Language Instruction."
                        },
                        "is_first": {
                            "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                            "tensor": {
                                "shape": {},
                                "dtype": "bool",
                                "encoding": "none"
                            },
                            "description": "True on first step of the episode."
                        },
                        "discount": {
                            "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                            "tensor": {
                                "shape": {},
                                "dtype": "float32",
                                "encoding": "none"
                            },
                            "description": "Discount if provided, default to 1."
                        },
                        "reward": {
                            "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                            "tensor": {
                                "shape": {},
                                "dtype": "float32",
                                "encoding": "none"
                            },
                            "description": "Reward if provided, 1 on final step for demos."
                        }
                    },
                    "length": "-1"
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Path to the original data file."
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Analyze each column to build the proper feature specification
    for col in df.columns:
        first_value = df[col].iloc[0]
        
        if col == 'action':
            # Update action shape
            if hasattr(first_value, 'shape'):
                features_dict["featuresDict"]["features"]["steps"]["feature"]["action"]["tensor"]["shape"]["dimensions"] = [str(d) for d in first_value.shape]
                
        elif col.startswith('observation.'):
            # Handle observation data
            obs_name = col.replace('observation.', '')
            
            if hasattr(first_value, 'shape') and len(first_value.shape) > 0:
                # Vector observation (like state)
                features_dict["featuresDict"]["features"]["steps"]["feature"]["observation"]["featuresDict"]["features"][obs_name] = {
                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                    "tensor": {
                        "shape": {"dimensions": [str(d) for d in first_value.shape]},
                        "dtype": "float32",
                        "encoding": "none"
                    },
                    "description": f"Robot {obs_name} observation."
                }
            else:
                # Scalar observation
                features_dict["featuresDict"]["features"]["steps"]["feature"]["observation"]["featuresDict"]["features"][obs_name] = {
                    "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", 
                    "tensor": {
                        "shape": {},
                        "dtype": "float32" if isinstance(first_value, (float, np.floating)) else "int64",
                        "encoding": "none"
                    },
                    "description": f"Robot {obs_name} observation."
                }
        
        elif col in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            # These are metadata, could be added as additional fields if needed
            pass
    
    # Add camera image features based on the info.json metadata
    camera_names = ["observation_images_top", "observation_images_hand", "observation_images_front"]
    for camera_name in camera_names:
        features_dict["featuresDict"]["features"]["steps"]["feature"]["observation"]["featuresDict"]["features"][camera_name] = {
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
            "image": {
                "shape": {"dimensions": ["224", "224", "3"]},  # Resized to 224x224
                "dtype": "uint8",
                "encoding": "png"
            },
            "description": f"Camera image: {camera_name}."
        }
    
    # Handle image observations if they exist (check for common image column patterns)
    potential_image_cols = [col for col in df.columns if 'image' in col.lower() or 'camera' in col.lower()]
    for col in potential_image_cols:
        first_value = df[col].iloc[0]
        if hasattr(first_value, 'shape') and len(first_value.shape) == 3:
            # This looks like an image
            img_name = col.replace('observation.', '').replace('.', '_')
            features_dict["featuresDict"]["features"]["steps"]["feature"]["observation"]["featuresDict"]["features"][img_name] = {
                "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                "image": {
                    "shape": {"dimensions": [str(d) for d in first_value.shape]},
                    "dtype": "uint8",
                    "encodingFormat": "jpeg"
                },
                "description": f"{img_name} camera RGB observation."
            }
    
    # Write the JSON files
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info_json, f, indent=2)

    with open(os.path.join(output_dir, "features.json"), 'w') as f:
        json.dump(features_dict, f, indent=4)
    
    print(f"Generated RLDS metadata files:")
    print(f"  - {output_dir}/dataset_info.json")
    print(f"  - {output_dir}/features.json")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to RLDS format")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of HuggingFace repository containing a LeRobotDataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the RLDS dataset. If not provided, uses the default TFDS data directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the dataset stored locally.",
    )
    
    args = parser.parse_args()
    # Load the dataset
    dataset = load_dataset_and_save_to_disk(args.repo_id, args.root, args.local_files_only, args.output_dir)
    

if __name__ == "__main__":
    main() 