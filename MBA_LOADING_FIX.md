# MBA Dataset Loading Fix

## Problem Identified
The previous code had a critical bug in `load_dataset()` that caused MBA to fail:

```python
# BAD CODE (line 104 in main.py)
else:
    print(f"📊 Detected multivariate time series data ({dataset})")
    print(f"   Keeping 2D format for overlapping windows: {train_data.shape}")
    # Don't reshape - keep as (time_steps, features)
    test_data = test_data[:, np.newaxis, :]  # ❌ BUG: This reshapes test_data!
```

This caused:
- Train data: `(64, 2)` - Only first batch due to DataLoader
- Test data: `(7680, 1, 2)` - Incorrectly reshaped to 3D
- Result: Dimension mismatch errors and incorrect windowing

## Fix Applied

### 1. Remove Incorrect test_data Reshape (main.py, line 103)
```python
# FIXED CODE
else:
    # MBA, SMAP, etc: Multivariate time series format: (time_steps, features)
    # KEEP AS 2D for overlapping window processing
    print(f"📊 Detected multivariate time series data ({dataset})")
    print(f"   Keeping 2D format for overlapping windows: {train_data.shape}")
    # Don't reshape - keep BOTH train and test as (time_steps, features)
    # ✅ No reshaping - both stay 2D
```

### 2. Use Full Batch for MBA (main.py, lines 111-118)
```python
# Use appropriate batch size
# For MBA/SMAP (2D multivariate): use full batch for overlapping windows
# For ECG (3D univariate): use smaller batches
if len(loader[0].shape) == 2:
    # MBA case: load full dataset as single batch for overlapping window processing
    batch_size = loader[0].shape[0]  # 7680 for MBA
else:
    # ECG case: use reasonable batch size
    batch_size = min(64, loader[0].shape[0])
```

## Expected Results After Fix

### Data Shapes
```
Training samples: torch.Size([7680, 2])  ✅ (was [64, 2])
Testing samples: torch.Size([7680, 2])   ✅ (was [7680, 1, 2])
Labels shape: (7680, 2)                  ✅
```

### Windowing
```
🔍 Processing 2D data: 7680 timesteps, 2 features
   Using OVERLAPPING windows (NO downsampling) for better accuracy
🎯 Created 7680 overlapping windows (all 7680 timesteps preserved)
```

### Training
```
📈 MBA-specific hyperparameters:
   Epochs: 50
   Learning rate: 0.005 (5e-3)  ✅ (not 0.001)
   Weight decay: 0.0001
   Note: Using overlapping windows (no downsampling) for maximum accuracy
```

## Why This Fix Works

1. **2D Format Preserved**: Both train and test data stay as `(7680, 2)` throughout
2. **Full Batch Loading**: All 7680 samples loaded at once for overlapping window creation
3. **Correct Windowing**: `convert_to_windows()` processes 2D data with overlapping strategy
4. **No Dimension Mismatches**: Model expects `[batch, features, window_size]` and gets it correctly

## Testing

After applying this fix, running:
```bash
python train_optimized_enhanced_dtaad.py --dataset MBA
```

Should produce:
- No dimension mismatch errors
- Proper overlapping window creation (7680 windows)
- Higher learning rate (5e-3) for better convergence
- Successful training and evaluation

## Files Modified

- `main.py`:
  - Line 103: Removed incorrect `test_data` reshape
  - Lines 111-118: Added conditional batch sizing logic
