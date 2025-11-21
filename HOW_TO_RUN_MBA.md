# MBA Dataset - How to Run and What to Expect

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install torch numpy pandas tqdm matplotlib scikit-learn scikit-plots seaborn openpyxl
   ```

2. **Preprocess MBA dataset:**
   ```bash
   python preprocess.py MBA
   ```

## Running the Training

```bash
python train_optimized_enhanced_dtaad.py --dataset MBA
```

## Expected Output

### ✅ Stage 1: Initialization
```
🚀 Training Optimized Enhanced DTAAD for MBA Anomaly Detection
============================================================
📊 Detected multivariate time series data (MBA)
   Keeping 2D format for overlapping windows: (7680, 2)
🔧 Using device: cpu
```

### ✅ Stage 2: Data Shapes (CRITICAL - Verify These!)
```
📊 Dataset Information:
  Training samples: torch.Size([7680, 2])  ← Must be 7680, not 64!
  Testing samples: torch.Size([7680, 2])   ← Must be 2D, not 3D!
  Number of features: 2
DEBUG: DTAAD init called with feats=2
```

### ✅ Stage 3: Hyperparameters
```
📈 MBA-specific hyperparameters:
  Epochs: 50
  Learning rate: 0.005 (increased for better F1)  ← Must be 0.005, not 0.001!
  Weight decay: 0.0001
  Note: Using overlapping windows (no downsampling) for maximum accuracy
```

### ✅ Stage 4: Windowing
```
🔍 Processing 2D data: 7680 timesteps, 2 features  ← Must say 7680!
   Using OVERLAPPING windows (NO downsampling) for better accuracy
🎯 Created 7680 overlapping windows (all 7680 timesteps preserved)  ← All preserved!
```

### ✅ Stage 5: Training Progress
```
📋 MBA: Using original labels (one-to-one with overlapping windows)

🏋️  Starting Optimized Training (50 epochs)...
Training Optimized Enhanced DTAAD:   0%|          | 0/50 [00:00<?, ?it/s]
Epoch 0, L1 = 0.02XXXX
Epoch 1, L1 = 0.01XXXX
...
Epoch 10/50, Loss: 0.01XXXX, LR: 0.004500  ← Learning rate decaying
...
Epoch 20/50, Loss: 0.01XXXX, LR: 0.004050
...
Epoch 30/50, Loss: 0.01XXXX, LR: 0.003645
...
Epoch 40/50, Loss: 0.01XXXX, LR: 0.003280
...
Epoch 50/50, Loss: 0.01XXXX, LR: 0.002952

Training Optimized Enhanced DTAAD: 100%|████████| 50/50 [00:XX<00:00, XX.XXit/s]
```

### ✅ Stage 6: Completion
```
⏱️  Optimized Training completed in X.XX seconds
💾 Optimized Enhanced model saved to: checkpoints/Optimized_Enhanced_DTAAD_MBA/model.ckpt

🧪 Testing Optimized Enhanced DTAAD...
Testing Optimized_Enhanced_DTAAD on MBA
✅ MBA: Labels already aligned (overlapping windows)
✅ Final shapes - loss: (7680, 2), labels: (7680, 2)

📊 Optimized Enhanced DTAAD Results (Per Feature):
         f1  precision    recall   TP     TN   FP   FN   ROC/AUC  threshold
0  0.XXXXX   0.XXXXX   0.XXXXX  XXX  XXXX  XXX  XXX  0.XXXXX   0.XXXXX
1  0.XXXXX   0.XXXXX   0.XXXXX  XXX  XXXX  XXX  XXX  0.XXXXX   0.XXXXX

🎯 Overall Performance on MBA Dataset:
{'FN': XX,
 'FP': XX,
 'f1': 0.XXXXX,  ← Target: > 0.97
 'precision': 0.XXXXX,
 'recall': 0.XXXXX,
 ...}

✅ Optimized Enhanced DTAAD training completed!
⏱️  Total time: XX.XX seconds
```

## Critical Checkpoints

When you run the code, **VERIFY THESE VALUES**:

| Checkpoint | Expected Value | Why Important |
|------------|---------------|---------------|
| Training samples | `torch.Size([7680, 2])` | Full dataset loaded |
| Testing samples | `torch.Size([7680, 2])` | Stays 2D (not 3D) |
| Learning rate | `0.005` | Updated value |
| Timesteps in windowing | `7680 timesteps` | All data preserved |
| Windows created | `7680 overlapping windows` | No downsampling |
| Final loss shapes | `(7680, 2)` | Matches labels |

## If You See Errors

### ❌ Error: `Training samples: torch.Size([64, 2])`
**Problem:** Only first batch loaded  
**Solution:** Make sure you pulled the latest commit (05e13b4 or later)

### ❌ Error: `Testing samples: torch.Size([7680, 1, 2])`
**Problem:** Test data incorrectly reshaped to 3D  
**Solution:** Line 104 bug - make sure it's removed in your version

### ❌ Error: `Learning rate: 0.001`
**Problem:** Old learning rate  
**Solution:** Make sure you have commit 1081bc3 or later

### ❌ Error: Dimension mismatch in training
**Problem:** Shapes don't match  
**Solution:** All fixes should be applied - pull latest commits

## Commits to Verify

Check your local branch has these commits:
```bash
git log --oneline -7
```

Should show:
- `3b446fc` Add verification script and output comparison documentation
- `05e13b4` Fix MBA dataset loading bug - remove incorrect test_data reshape
- `1081bc3` Increase MBA learning rate to 5e-3 and clarify no downsampling
- `5cd72fb` Add implementation summary and document current status
- `7265a05` Add overlapping windows support and improve hyperparameters for MBA
- `0736a6b` Fix multivariate data compatibility for Optimized Enhanced DTAAD
- `2514efa` Initial plan

## Performance Expectations

With these fixes:
- **Training**: ~3-5 seconds for 50 epochs (MBA is small)
- **F1 Score**: Should improve toward >0.97 target
- **No errors**: Should complete without dimension mismatches
- **All data used**: 7680 windows = 7680 timesteps (no loss)

## Troubleshooting

If issues persist:
1. Delete `processed/MBA/` folder
2. Re-run `python preprocess.py MBA`
3. Make sure you're on the latest commit
4. Check that main.py line 104 does NOT have `test_data = test_data[:, np.newaxis, :]`
