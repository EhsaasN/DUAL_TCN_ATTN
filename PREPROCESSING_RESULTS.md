# Preprocessing Results - ECG and MBA Datasets

## Summary

Both ECG and MBA datasets have been successfully preprocessed and are ready for training with `train_optimized_enhanced_dtaad.py`.

---

## ECG Data Results

### Raw Data Shapes
- **Train**: (190, 17479) - 190 samples, 17479 timesteps each
- **Test**: (48, 17479) - 48 samples, 17479 timesteps each  
- **Labels**: (48, 1) - 48 labels (one per test sample)
- **Total anomalies**: 31 out of 48 samples (64.6%)

### Dataset Characteristics
- **Type**: Univariate (ECG) - single sensor reading
- **Windowing Strategy**: Non-overlapping (step=window_size=10)
- **Reason**: Memory optimization for large sequences (17479 timesteps per sample)

### Expected Processing
- **Windows per sample**: 1747 (17479 ÷ 10)
- **Total windows**: 83,856 (48 samples × 1747 windows)
- **Label handling**: Labels will be repeated for each window (31 anomalous samples → 54,357 anomalous windows)

### Data Statistics
- Train: mean=0.4497, std=0.1758, range=[0.0000, 1.0000]
- Test: mean=0.4469, std=0.1764, range=[0.0000, 1.0000]
- Data is normalized to [0, 1] range

### Expected Performance (from main branch)
- **F1 Score**: ~0.9999
- **False Positives (FP)**: ~30
- **False Negatives (FN)**: ~1
- **Precision**: ~0.9995
- **Recall**: ~1.0000

---

## MBA Data Results

### Raw Data Shapes
- **Train**: (7680, 2) - 7680 timesteps, 2 features
- **Test**: (7680, 2) - 7680 timesteps, 2 features
- **Labels**: (7680, 2) - 7680 labels (one per timestep, per feature)
- **Total anomalies**: 5200 (2600 per feature)

### Dataset Characteristics
- **Type**: Multivariate - 2 sensor readings
- **Windowing Strategy**: Overlapping (one window per timestep)
- **Reason**: Preserve all temporal information for better accuracy

### Expected Processing
- **Total windows**: 7680 (overlapping - each timestep creates a window)
- **Label handling**: One-to-one mapping (each window has its corresponding label)

### Data Statistics
- Train: mean=0.5501, std=0.1093, range=[0.0003, 1.0013]
- Test: mean=0.5483, std=0.1087, range=[-0.0013, 0.9993]
- Data is approximately normalized to [0, 1] range

### Anomaly Distribution
- **Samples with anomalies**: 2600/7680 (33.9%)
- **Feature 1 anomalies**: 2600
- **Feature 2 anomalies**: 2600
- Both features have identical anomaly patterns

### Expected Performance (target)
- **F1 Score**: > 0.97
- **Training time**: ~3-5 seconds (50 epochs)
- **Configuration**: 
  - Learning rate: 5e-3 (5x higher than default)
  - Epochs: 50
  - No downsampling (all 7680 timesteps preserved)

---

## Key Differences

| Aspect | ECG | MBA |
|--------|-----|-----|
| **Data Type** | Univariate (1 feature) | Multivariate (2 features) |
| **Structure** | Multiple samples (48) | Single long sequence (7680 timesteps) |
| **Windowing** | Non-overlapping | Overlapping |
| **Windows Created** | 83,856 (1747 per sample) | 7,680 (one per timestep) |
| **Label Strategy** | Repeat per window | One-to-one mapping |
| **Memory Usage** | High (large sequences) | Low (small dataset) |
| **Training Time** | ~30-60 seconds | ~3-5 seconds |

---

## Windowing Explanation

### ECG (Non-overlapping)
```
Original: [Sample 1: 17479 timesteps] → Windows of size 10, step 10
Result: 1747 windows from Sample 1
        Window 1: timesteps 0-9
        Window 2: timesteps 10-19
        ...
        Window 1747: timesteps 17470-17479
```

**Label Aggregation**: Each of the 1747 windows gets the same label as the original sample. If sample is anomalous, all 1747 windows are labeled anomalous.

### MBA (Overlapping)
```
Original: [7680 timesteps with 2 features] → Windows of size 10, step 1
Result: 7680 windows (sliding window)
        Window 1: timesteps 0-9
        Window 2: timesteps 1-10
        Window 3: timesteps 2-11
        ...
        Window 7680: timesteps 7671-7680
```

**Label Mapping**: Each window gets the label of its last timestep. Window i gets label from timestep i.

---

## Verification Commands

```bash
# Check preprocessing output
python test_preprocessing_output.py

# Train on ECG
python train_optimized_enhanced_dtaad.py --dataset ecg_data

# Train on MBA  
python train_optimized_enhanced_dtaad.py --dataset MBA
```

---

## Files Created

### ECG Data
- `processed/ecg_data/train.npy` - 26 MB
- `processed/ecg_data/test.npy` - 6.5 MB
- `processed/ecg_data/labels.npy` - 512 bytes

### MBA Data
- `processed/MBA/train.npy` - 121 KB
- `processed/MBA/test.npy` - 121 KB
- `processed/MBA/labels.npy` - 121 KB

---

## Next Steps

1. ✅ **Preprocessing Complete**: Both datasets ready
2. ⏳ **Training**: Run training scripts to validate performance
3. ⏳ **Validation**: Verify FP/FN metrics match expectations
   - ECG: Should achieve ~30 FP, ~1 FN (F1 ~0.9999)
   - MBA: Should achieve F1 > 0.97

---

## Notes

- Both datasets are normalized and ready for training
- ECG uses non-overlapping windows to prevent memory crashes (as requested by user)
- MBA uses overlapping windows for better accuracy (no data loss)
- Label aggregation for ECG has been fixed to prevent the 500 FP issue
- All code fixes are in place and syntax errors have been resolved
