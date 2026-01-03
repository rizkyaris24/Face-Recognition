# Download Age and Gender Detection Models

The age and gender detection models (`.caffemodel` files) are large (~41MB each) and need to be downloaded separately.

## Quick Download Instructions

### Option 1: Automated Download Script (Recommended)

Run the download script:
```bash
python download_models.py
```

This will attempt to download the models automatically.

### Option 2: Manual Download

If the automated script doesn't work, download manually:

#### Step 1: Download Age Model

1. Visit the research paper page:
   - **URL**: https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.html
   - Look for the "Age Estimation" model download link
   - Download: `age_net.caffemodel` (~41MB)

2. Alternative sources:
   - Search GitHub for: `age_net.caffemodel opencv`
   - Try: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

#### Step 2: Download Gender Model

1. From the same research page:
   - **URL**: https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.html
   - Look for the "Gender Classification" model download link
   - Download: `gender_net.caffemodel` (~41MB)

2. Alternative sources:
   - Search GitHub for: `gender_net.caffemodel opencv`
   - Try: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

#### Step 3: Place Files

Move the downloaded files to your `models/` directory:
```
models/
├── age_deploy.prototxt          ✅ (already exists)
├── age_net.caffemodel           ⬇️  (download required)
├── gender_deploy.prototxt       ✅ (already exists)
└── gender_net.caffemodel        ⬇️  (download required)
```

## Verification

After downloading, verify the files:

1. Check file sizes:
   - `age_net.caffemodel`: Should be ~41MB
   - `gender_net.caffemodel`: Should be ~41MB

2. Run the app:
   ```bash
   python app.py
   ```

3. Look for these messages:
   ```
   ✅ Age detection enabled
   ✅ Gender detection enabled
   ```

## Direct Download Links (May Change)

If you have direct access to these repositories, try:

- OpenCV Extra Test Data: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
- Research Paper Repository: Check the paper's official page for updated links

## Troubleshooting

### "Model files not found"
- Ensure files are in the `models/` directory
- Check file names match exactly (case-sensitive)
- Verify file sizes are ~41MB (not corrupted)

### "Error loading age/gender model"
- Files might be corrupted - re-download
- Ensure both `.prototxt` and `.caffemodel` files are present
- Check OpenCV version compatibility

### Download Issues
- Use a download manager for large files
- Try different browsers
- Use VPN if downloads are blocked
- Check university/research institution mirrors

## Current Status

Your application works without these models, but will only show:
- ✅ Face detection
- ✅ Face counting
- ✅ Distance estimation
- ❌ Age prediction (requires models)
- ❌ Gender prediction (requires models)

After downloading the models, all features will be enabled!

