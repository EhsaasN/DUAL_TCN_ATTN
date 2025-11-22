# MBA Dataset Configuration Changes

## Summary of Changes (Per User Request)

### 1. Increased Learning Rate ✅
- **Previous**: `learning_rate = 1e-3` (0.001)
- **Updated**: `learning_rate = 5e-3` (0.005)
- **Impact**: 5x increase for faster convergence and potentially better F1 score

### 2. No Downsampling ✅
- **Implementation**: Overlapping windows (already implemented)
- **Details**: 
  - All 7680 MBA timesteps are preserved
  - Each timestep gets its own window
  - No data is skipped or downsampled
- **Impact**: Maximum pattern recognition and accuracy

## Configuration Details

### MBA Hyperparameters
```python
num_epochs = 50
learning_rate = 5e-3  # INCREASED (was 1e-3)
weight_decay = 1e-4
step_size = 10
gamma = 0.9
optimizer = AdamW
scheduler = StepLR
```

### Windowing Strategy
```python
# MBA uses overlapping windows - NO downsampling
for i in range(len(data)):  # All 7680 timesteps
    if i >= w_size:
        w = data[i - w_size:i]  # Overlapping window
    else:
        w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
    windows.append(w)

# Result: 7680 windows for 7680 timesteps (one-to-one)
```

### Comparison: ECG vs MBA

| Aspect | ECG (Univariate) | MBA (Multivariate) |
|--------|------------------|-------------------|
| Data Shape | (samples, 1, sequence) | (timesteps, features) |
| Windowing | Non-overlapping | **Overlapping** |
| Step Size | window_size (downsampled) | 1 (NO downsampling) |
| Windows Created | ~1747 from 17479 steps | 7680 from 7680 steps |
| Learning Rate | 4e-4 | **5e-3** (12.5x higher) |
| Epochs | 5 | 50 |
| Purpose | Memory optimization | **Accuracy optimization** |

## Expected Results

With these changes:
1. **Higher LR (5e-3)**: Faster convergence, better optimization landscape exploration
2. **No Downsampling**: Complete temporal information preserved
3. **50 Epochs**: Sufficient training for convergence

### Target Performance
- **F1 Score**: Should improve toward >0.97 target
- **Training Time**: ~4-8 seconds (MBA is small dataset)
- **Pattern Recognition**: Better due to complete temporal coverage

## Files Modified

1. **train_optimized_enhanced_dtaad.py**
   - Line 131: Updated `learning_rate = 5e-3` (was `1e-3`)
   - Line 137: Added note about increased LR
   - Line 138: Added note about no downsampling

2. **main.py**
   - Line 55: Added comment "NO DOWNSAMPLING"
   - Line 57: Updated print message to clarify "NO downsampling"
   - Line 69: Enhanced print to show all timesteps preserved

## Code Locations

### Learning Rate Change
```python
# File: train_optimized_enhanced_dtaad.py, Line 131
if dataset == 'MBA':
    learning_rate = 5e-3  # Increased from 1e-3
```

### No Downsampling Confirmation
```python
# File: main.py, Lines 59-69
for i in range(len(data)):  # Processes ALL timesteps
    if i >= w_size:
        w = data[i - w_size:i]  # Overlapping, not skipping
```

## Testing

To test the changes:
```bash
python3 train_optimized_enhanced_dtaad.py --dataset MBA
```

Expected output will show:
- Learning rate: 0.005 (5e-3)
- "Using OVERLAPPING windows (NO downsampling)"
- "Created 7680 overlapping windows (all 7680 timesteps preserved)"

## Rationale

### Why Higher Learning Rate?
- MBA has only 7680 samples (small dataset)
- Higher LR helps escape local minima faster
- With AdamW's adaptive learning, 5e-3 is safe and effective

### Why No Downsampling?
- MBA temporal patterns are important
- Overlapping windows capture all transitions
- Original DTAAD paper used this approach for small datasets
- Maximum information = better anomaly detection
