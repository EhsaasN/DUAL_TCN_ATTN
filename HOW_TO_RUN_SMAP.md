# How to Run SMAP Dataset

## Understanding SMAP Data Structure

SMAP (and MSL) datasets are **different** from other datasets. After preprocessing:
- **Other datasets** create: `train.npy`, `test.npy`, `labels.npy`
- **SMAP/MSL** create: `{channel}_train.npy`, `{channel}_test.npy`, `{channel}_labels.npy` for each channel

Example SMAP preprocessed files:
```
processed/SMAP/
  P-1_train.npy
  P-1_test.npy
  P-1_labels.npy
  P-2_train.npy
  P-2_test.npy
  P-2_labels.npy
  ... (multiple channels)
```

## How to Run SMAP

### Step 1: Preprocess SMAP Data

```bash
python preprocess.py SMAP
```

This will create multiple channel files in `processed/SMAP/` folder.

### Step 2: Run Training

```bash
python train_optimized_enhanced_dtaad.py --dataset SMAP
```

The code will **automatically load the first available channel**.

## Expected Output

```
🎯 Starting training for dataset: SMAP
🚀 Training Optimized Enhanced DTAAD for SMAP Anomaly Detection
============================================================
📊 Loading SMAP channel: P-1
   Available channels: 55
📊 Detected multivariate time series data (SMAP)
   Keeping 2D format for overlapping windows: (X, 25)

📊 Dataset Information:
  Training samples: torch.Size([X, 25])   ✅
  Testing samples: torch.Size([Y, 25])    ✅
  Number of features: 25

🔍 Processing 2D data: X timesteps, 25 features
   Using OVERLAPPING windows (NO downsampling) for better accuracy
🎯 Created X overlapping windows (all timesteps preserved)

📋 SMAP: Using original labels (one-to-one with overlapping windows)

🏋️  Starting Optimized Training...
[Training proceeds successfully]
```

## Important Notes

1. **Multiple Channels**: SMAP has ~55 channels, the code loads the first one by default
2. **File Pattern**: Files are named `{channel}_train.npy` not `train.npy`
3. **Auto-Detection**: The code automatically detects and loads the first channel
4. **Same Processing**: SMAP uses the same multivariate processing as MBA

## Training Different SMAP Channels

If you want to train on a specific SMAP channel, you would need to modify the code or create symbolic links:

```bash
# Option 1: Create standard file names (temporary workaround)
cd processed/SMAP
ln -s P-1_train.npy train.npy
ln -s P-1_test.npy test.npy  
ln -s P-1_labels.npy labels.npy
```

But this is not recommended - the code now handles it automatically.

## Troubleshooting

### Error: "No processed data files found for SMAP"

This means preprocessing didn't work or the files weren't created.

**Check:**
```bash
ls processed/SMAP/
```

You should see files like: `P-1_train.npy`, `P-1_test.npy`, etc.

If folder is empty:
1. Check if raw SMAP data exists in `data/SMAP_MSL/`
2. Re-run preprocessing: `python preprocess.py SMAP`

### Error: "Processed data not found for SMAP"

The `processed/SMAP/` folder doesn't exist.

**Solution:**
```bash
python preprocess.py SMAP
```

## Quick Test with MBA (Simpler)

If SMAP is giving issues, test with MBA first (simpler structure):

```bash
python preprocess.py MBA
python train_optimized_enhanced_dtaad.py --dataset MBA
```

MBA creates standard files: `train.npy`, `test.npy`, `labels.npy`

## Summary

✅ **SMAP is now fully supported** with automatic channel detection
- Handles the special file naming pattern `{channel}_train.npy`
- Automatically loads the first available channel  
- Works exactly like MBA for multivariate processing
- No manual intervention needed

The fix in commit handles the SMAP-specific file structure automatically!
