#!/usr/bin/env python3
"""
Download script for Age and Gender Detection models.
Downloads the required .caffemodel files for age and gender prediction.
"""

import os
import urllib.request
import subprocess
import sys
from pathlib import Path

# Model URLs - using reliable sources
MODEL_URLS = {
    'age_net.caffemodel': 'https://drive.google.com/uc?export=download&id=1_ht2s4bV1mkjY9Wdr8RiO9zYwF0F2hg_',
    'gender_net.caffemodel': 'https://drive.google.com/uc?export=download&id=1W_moLzMlGiELyPxwiYwPUX8T_SYRfgeP',
}

# Alternative URLs if Google Drive doesn't work
ALTERNATIVE_URLS = {
    'age_net.caffemodel': [
        # Try OpenCV samples repository (more likely to work)
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/age_net.caffemodel',
        'https://github.com/opencv/opencv/raw/master/samples/dnn/age_net.caffemodel',
        # Try alternative GitHub repos
        'https://github.com/akshtsng/Gender_Detection_and_Age_Prediction/raw/master/age_net.caffemodel',
    ],
    'gender_net.caffemodel': [
        # Try OpenCV samples repository (more likely to work)
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/gender_net.caffemodel',
        'https://github.com/opencv/opencv/raw/master/samples/dnn/gender_net.caffemodel',
        # Try alternative GitHub repos
        'https://github.com/akshtsng/Gender_Detection_and_Age_Prediction/raw/master/gender_net.caffemodel',
    ],
}

def download_file(url, dest_path, model_name):
    """Download a file with progress indicator."""
    try:
        print(f"📥 Downloading {model_name} from {url[:60]}...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print("\n   ✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return False

def download_with_curl(url, dest_path, model_name):
    """Try downloading with curl as fallback."""
    try:
        print(f"📥 Trying curl to download {model_name}...")
        result = subprocess.run(
            ['curl', '-L', '--progress-bar', '--output', str(dest_path), url],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000000:  # > 1MB
            print("   ✅ Download complete!")
            return True
        else:
            if dest_path.exists():
                dest_path.unlink()  # Remove incomplete file
            return False
    except Exception as e:
        print(f"   ❌ Curl error: {e}")
        return False

def download_models():
    """Download age and gender detection models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🔍 Checking for missing model files...\n")
    
    models_to_download = []
    for model_name in ['age_net.caffemodel', 'gender_net.caffemodel']:
        model_path = models_dir / model_name
        if not model_path.exists():
            models_to_download.append(model_name)
            print(f"❌ Missing: {model_name}")
        else:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb < 1:  # Model should be ~41MB, if < 1MB it's likely corrupted
                print(f"⚠️  Found but suspiciously small ({size_mb:.1f}MB): {model_name}")
                models_to_download.append(model_name)
            else:
                print(f"✅ Already exists: {model_name} ({size_mb:.1f}MB)")
    
    if not models_to_download:
        print("\n✅ All model files are present!")
        return
    
    print(f"\n📦 Need to download {len(models_to_download)} model(s)\n")
    
    for model_name in models_to_download:
        model_path = models_dir / model_name
        print(f"\n{'='*60}")
        print(f"Downloading: {model_name}")
        print(f"{'='*60}")
        
        # Try primary URL first
        url = MODEL_URLS.get(model_name)
        success = False
        
        if url:
            success = download_file(url, model_path, model_name)
        
        # If primary fails, try alternatives
        if not success and model_name in ALTERNATIVE_URLS:
            print(f"\n   Trying alternative sources...")
            for alt_url in ALTERNATIVE_URLS[model_name]:
                success = download_file(alt_url, model_path, model_name)
                if success:
                    break
                # Try curl as fallback
                if not success:
                    success = download_with_curl(alt_url, model_path, model_name)
                    if success:
                        break
        
        if not success:
            print(f"\n⚠️  Could not download {model_name} automatically.")
            print(f"   Please download manually from:")
            print(f"   - Visit: https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.html")
            print(f"   - Or search GitHub for: '{model_name}' opencv")
            print(f"   - Save to: {model_path.absolute()}\n")
        else:
            # Verify file size
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.1f}MB")
            if size_mb < 5:
                print(f"   ⚠️  Warning: File seems too small. Expected ~41MB.")
                print(f"   The file might be corrupted. Please verify manually.")
    
    print(f"\n{'='*60}")
    print("Download process complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    download_models()
    print("\n💡 Tip: If downloads failed, you can also:")
    print("   1. Visit: https://talhassner.github.io/home/projects/cnn_agegender/")
    print("   2. Download the models manually")
    print("   3. Place them in the 'models/' directory")

