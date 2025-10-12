import tensorflow as tf
import json

# Path to your TFRecord file and features.json
tfrecord_path = "/home/hhjiang/Documents/VLA-Adapter/data/lerobot/black_object_test_rlds/black_object_test/1.0.0/black_object_test-train.tfrecord-00000-of-00002"
features_json_path = "/home/hhjiang/Documents/VLA-Adapter/data/lerobot/black_object_test_rlds/black_object_test/1.0.0/features.json"

# Read features.json
with open(features_json_path, "r") as f:
    features = json.load(f)

def print_features_dict(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict) and "dtype" not in v:
            print_features_dict(v, prefix + k + ".")
        else:
            print(f"features.json: {prefix}{k} -> {v.get('dtype', 'unknown')} shape={v.get('shape', 'unknown')}")

print("=== features.json fields ===")
steps = features.get("features", {}).get("steps", {}).get("feature", {})
features_keys = set()
def collect_keys(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict) and "dtype" not in v:
            collect_keys(v, prefix + k + ".")
        else:
            features_keys.add(prefix + k)
            print(f"features.json: {prefix}{k} -> {v.get('dtype', 'unknown')} shape={v.get('shape', 'unknown')}")
collect_keys(steps)

# Read one example from TFRecord
raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
tfrecord_keys = set()
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print("\n=== TFRecord keys and types ===")
    for key, feature in example.features.feature.items():
        tfrecord_keys.add(key)
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            dtype = "bytes"
            value = feature.bytes_list.value
        elif kind == "float_list":
            dtype = "float"
            value = feature.float_list.value
        elif kind == "int64_list":
            dtype = "int64"
            value = feature.int64_list.value
        else:
            dtype = "unknown"
            value = None
        print(f"TFRecord: {key} -> {dtype} (len={len(value) if value is not None else 'N/A'})")
    break

# Highlight mismatches
print("\n=== Key Comparison ===")
missing_in_tfrecord = features_keys - tfrecord_keys
extra_in_tfrecord = tfrecord_keys - features_keys
if missing_in_tfrecord:
    print("Keys in features.json but missing in TFRecord:")
    for k in missing_in_tfrecord:
        print(f"  {k}")
else:
    print("No missing keys in TFRecord.")
if extra_in_tfrecord:
    print("Keys in TFRecord but not in features.json:")
    for k in extra_in_tfrecord:
        print(f"  {k}")
else:
    print("No extra keys in TFRecord.")