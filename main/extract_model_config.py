import h5py
import json
import os
from zipfile import ZipFile
import argparse

def extract_config(model_path):
    # Try HDF5 format first
    try:
        with h5py.File(model_path, 'r') as f:
            config = json.loads(f.attrs['model_config'].decode('utf-8'))
            save_config(config, model_path)
            return True
    except Exception as e:
        print(f"HDF5 extraction failed: {e}")
    
    # Try new ZIP format
    try:
        with ZipFile(model_path, 'r') as z:
            with z.open('config.json') as f:
                config = json.load(f)
            save_config(config, model_path)
            return True
    except Exception as e:
        print(f"ZIP format extraction failed: {e}")
    
    return False

def save_config(config, model_path):
    output_path = f"{model_path}.config.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .keras model file')
    args = parser.parse_args()
    
    if not extract_config(args.model_path):
        print("❌ All extraction attempts failed")
        print("Possible solutions:")
        print("1. Verify file is not corrupted")
        print("2. Try re-saving the model with:")
        print("   model.save('new_model.keras', save_format='keras')")