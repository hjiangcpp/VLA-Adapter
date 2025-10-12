"""
convert_lerobot_to_rlds.py

将LeRobot格式的数据转换为RLDS格式，类似于LIBERO数据集的格式。
"""


import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def convert_lerobot_to_rlds(lerobot_data_dir: str, target_dir: str, dataset_name: str = "lerobot_dataset_name"):
    """
    将LeRobot格式的数据转换为RLDS格式
    
    Args:
        lerobot_data_dir: LeRobot数据目录路径
        target_dir: 目标RLDS格式数据目录路径
        dataset_name: 数据集名称
    """
    print(f"Converting LeRobot data from {lerobot_data_dir} to RLDS format at {target_dir}")
    
    # 创建TFDS构建器目录结构
    os.makedirs(target_dir, exist_ok=True)
    builder_dir = os.path.join(target_dir, dataset_name, "1.0.0")
    os.makedirs(builder_dir, exist_ok=True)
    version_dir = builder_dir
    
    # 读取LeRobot元数据
    info_path = os.path.join(lerobot_data_dir, "meta", "info.json")
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # 读取episodes信息
    episodes_path = os.path.join(lerobot_data_dir, "meta", "episodes.jsonl")
    episodes = []
    length = info['total_episodes']
    with open(episodes_path, 'r') as f:
        for i in range(length):
            ep_path = os.path.join(lerobot_data_dir, "data", f"chunk-000/episode_{i:06d}.parquet")
            df = pd.read_parquet(ep_path)
            episodes.append({
                "episode_id": i,
                "length": len(df),
                "language_instruction": info.get('language_instructions', {}).get(str(i), "")
            })
            row = df.iloc[i]
            def process_image(img_arr):
                img = Image.fromarray(img_arr.astype(np.uint8))
                img = img.resize((256,256), Image.BILINEAR)
                return np.array(img)
            def decode_if_bytes(img):
                if isinstance(img, bytes):
                    arr = np.array(Image.open(io.BytesIO(img)))
                    return arr
                return img
            front_img = decode_if_bytes(row.get('observation.images.front', None))
            hand_img = decode_if_bytes(row.get('observation.images.hand', None))
            top_img = decode_if_bytes(row.get('observation.images.top', None))
            if front_img is not None:
                front_img = process_image(front_img)
            if hand_img is not None:
                hand_img = process_image(hand_img)
            if top_img is not None:
                top_img = process_image(top_img)
            def encode_jpeg(img):
                if img is None:
                    return b''
                img_pil = Image.fromarray(img)
                buf = io.BytesIO()
                img_pil.save(buf, format='JPEG')
                return buf.getvalue()
            # Build Libero-style nested step
            steps = {
                "feature": {
                    "observation": {
                        "front_image": encode_jpeg(front_img),
                        "hand_image": encode_jpeg(hand_img),
                        "top_image": encode_jpeg(top_img),
                        "state": np.array(row.get('observation.state', np.zeros(6, dtype=np.float32)), dtype=np.float32).flatten()
                    },
                    "action": np.array(row.get('action', np.zeros(6, dtype=np.float32)), dtype=np.float32).flatten(),
                    "reward": 1.0 if i == length-1 else 0.0,
                    "discount": 1.0,
                    "is_first": i == 0,
                    "is_last": i == length-1,
                    "is_terminal": i == length-1,
                    "language_instruction": episodes[-1]["language_instruction"] if episodes else ""
                },
                "length": -1
            }
            steps.append(step)
        ep_path = os.path.join(lerobot_data_dir, "data", f"chunk-000/episode_{episode_idx:06d}.parquet")
        if not os.path.exists(ep_path):
            print(f"Warning: missing {ep_path}")
          
        df = pd.read_parquet(ep_path)
        length = len(df)
        vla_features_format = {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {
                "features": {
                    "steps": {
                        "feature": {
                            "observation": {
                                "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                "featuresDict": {
                                    "features": {
                                        "front_image": {
                                            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                            "image": {
                                                "shape": {"dimensions": ["256", "256", "3"]},
                                                "dtype": "uint8",
                                                "encodingFormat": "jpeg"
                                            },
                                            "description": "Front camera RGB observation."
                                        },
                                        "hand_image": {
                                            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                            "image": {
                                                "shape": {"dimensions": ["256", "256", "3"]},
                                                "dtype": "uint8",
                                                "encodingFormat": "jpeg"
                                            },
                                            "description": "Hand camera RGB observation."
                                        },
                                        "top_image": {
                                            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                            "image": {
                                                "shape": {"dimensions": ["256", "256", "3"]},
                                                "dtype": "uint8",
                                                "encodingFormat": "jpeg"
                                            },
                                            "description": "Top camera RGB observation."
                                        },
                                        "state": {
                                            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                            "tensor": {
                                                "shape": {"dimensions": ["6"]},
                                                "dtype": "float32",
                                                "encoding": "none"
                                            },
                                            "description": "Robot joint angles."
                                        }
                                    }
                                }
                            },
                            "action": {
                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                "tensor": {
                                    "shape": {"dimensions": ["6"]},
                                    "dtype": "float32",
                                    "encoding": "none"
                                },
                                "description": "Robot action."
                            },
                            "reward": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {
                                    "shape": {},
                                    "dtype": "float32",
                                    "encoding": "none"
                                },
                                "description": "Reward if provided, 1 on final step for demos."
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
                            "is_last": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {
                                    "shape": {},
                                    "dtype": "bool",
                                    "encoding": "none"
                                },
                                "description": "True on last step of the episode."
                            },
                            "is_terminal": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {
                                    "shape": {},
                                    "dtype": "bool",
                                    "encoding": "none"
                                },
                                "description": "True on last step of the episode if it is a terminal step, True for demos."
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
                            "language_instruction": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Language Instruction."
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
    
        # 写入TFRecord文件
        num_shards = 2
        shard_size = (len(steps) + num_shards - 1) // num_shards
        shard_lengths = []
        for shard_id in range(num_shards):
            shard_path = os.path.join(version_dir, f"{dataset_name}-train.tfrecord-{shard_id:05d}-of-{num_shards:05d}")
            start_idx = shard_id * shard_size
            end_idx = min((shard_id + 1) * shard_size, len(steps))
            shard_steps = steps[start_idx:end_idx]
            shard_lengths.append(len(shard_steps))
            with tf.io.TFRecordWriter(shard_path) as writer:
                for step in tqdm(shard_steps, desc=f"Writing shard {shard_id}"):
                    example = tf.train.Example()
                    feature = example.features.feature
                    def _bytes_feature(value):
                        if isinstance(value, type(tf.constant(0))):  # if value is tensor
                            value = value.numpy()
                        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
                    def _float_feature(value):
                        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten() if isinstance(value, np.ndarray) else [value]))
                    def _int64_feature(value):
                        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten() if isinstance(value, np.ndarray) else [value]))
                    obs = step["feature"]["observation"]
                    feature["observation.front_image"].CopyFrom(_bytes_feature(obs["front_image"]))
                    feature["observation.hand_image"].CopyFrom(_bytes_feature(obs["hand_image"]))
                    feature["observation.top_image"].CopyFrom(_bytes_feature(obs["top_image"]))
                    feature["observation.state"].CopyFrom(_float_feature(obs["state"]))
                    feature["action"].CopyFrom(_float_feature(step["feature"]["action"]))
                    feature["reward"].CopyFrom(_float_feature(step["feature"]["reward"]))
                    feature["discount"].CopyFrom(_float_feature(step["feature"]["discount"]))
                    feature["is_first"].CopyFrom(_int64_feature([1 if step["feature"]["is_first"] else 0]))
                    feature["is_last"].CopyFrom(_int64_feature([1 if step["feature"]["is_last"] else 0]))
                    feature["is_terminal"].CopyFrom(_int64_feature([1 if step["feature"]["is_terminal"] else 0]))
                    feature["language_instruction"].CopyFrom(_bytes_feature(step["feature"]["language_instruction"].encode('utf-8')))
                    example_bytes = example.SerializeToString()
                    writer.write(example_bytes)
    
    with open(features_path, 'w') as f:
        json.dump(vla_features_format, f, indent=2)
    
    print(f"Conversion complete! Created {num_shards} shards with lengths: {shard_lengths}")
    print(f"Saved RLDS dataset at: {target_dir}")

if __name__ == "__main__":
    # 示例用法
    lerobot_data_dir = "/home/hhjiang/Documents/VLA-Adapter/data/lerobot/black-object-test-082525"
    target_dir = "/home/hhjiang/Documents/VLA-Adapter/data/lerobot/black_object_test_rlds"
    
    convert_lerobot_to_rlds(lerobot_data_dir, target_dir, "black_object_test")
