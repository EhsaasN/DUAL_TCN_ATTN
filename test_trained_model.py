#!/usr/bin/env python
"""
Standalone Model Testing Script for Optimized Enhanced DTAAD
Simply change the MODEL_PATH and DATASET variables to test different models/datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
import pandas as pd

# Import required modules
from src.enhanced_dtaad_optimized import OptimizedEnhancedDTAAD
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from main import convert_to_windows, load_dataset, backprop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES TO TEST DIFFERENT MODELS/DATASETS
# ============================================================================

# Path to the trained model checkpoint
MODEL_PATH = "checkpoints/Optimized_Enhanced_DTAAD_ecg_data/model.ckpt"
# Alternative for MBA: "checkpoints/Optimized_Enhanced_DTAAD_MBA/model.ckpt"

# Dataset to test on
DATASET = "ecg_data"  # Options: "ecg_data", "MBA", "SMAP", etc.

# ============================================================================

def test_model(model_path, dataset_name):
    """
    Test a trained model on specified dataset
    
    Args:
        model_path: Path to model checkpoint (.ckpt file)
        dataset_name: Name of dataset to test on
    """
    print("="*60)
    print(f"🧪 Testing Optimized Enhanced DTAAD Model")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {dataset_name}")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path):
                    model_file = os.path.join(item_path, "model.ckpt")
                    if os.path.exists(model_file):
                        print(f"  ✓ {model_file}")
        return None
    
    # Load dataset
    print(f"\n📊 Loading {dataset_name} dataset...")
    try:
        train_loader, test_loader, labels = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"\nMake sure to preprocess first: python preprocess.py {dataset_name}")
        return None
    
    # Load FULL dataset in one batch (same as training script)
    # This ensures we get all samples, not just first batch
    train_data_list = []
    for batch in train_loader:
        train_data_list.append(batch)
    if len(train_data_list) == 1:
        trainD = train_data_list[0]
    else:
        trainD = torch.cat(train_data_list, dim=0)
    
    test_data_list = []
    for batch in test_loader:
        test_data_list.append(batch)
    if len(test_data_list) == 1:
        testD = test_data_list[0]
    else:
        testD = torch.cat(test_data_list, dim=0)
    
    trainO, testO = trainD, testD
    
    print(f"  ✅ Loaded full dataset: train={trainD.shape}, test={testD.shape}")
    
    # Get data info
    if isinstance(testO, torch.Tensor):
        testO_np = testO.numpy()
    else:
        testO_np = testO
    
    num_features = labels.shape[1] if len(labels.shape) > 1 else 1
    
    print(f"\n📊 Dataset Information:")
    print(f"  Training shape: {trainD.shape}")
    print(f"  Testing shape: {testD.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of features: {num_features}")
    print(f"  Total test samples: {testD.shape[0]}")
    
    # Create model
    print(f"\n🏗️  Creating model architecture...")
    model = OptimizedEnhancedDTAAD(num_features).double()
    
    # Load trained weights
    print(f"📥 Loading trained weights from checkpoint...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # PyTorch 2.6+ requires weights_only=False for backward compatibility
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    try:
        # Load model state - note: dynamic decoders will be recreated on first forward pass
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model state: {e}")
        print(f"\nCheckpoint keys: {checkpoint.keys()}")
        return None
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"✅ Model loaded successfully!")
    print(f"   Training epoch: {epoch}")
    print(f"   Note: Dynamic decoders will be recreated on first forward pass")
    
    # Create optimizer and scheduler (needed for backprop function signature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("   Note: Could not load optimizer state (expected for dynamic decoders)")
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("   Note: Could not load scheduler state")
    
    # Convert to windows
    print(f"\n🪟 Converting data to windows...")
    try:
        trainD = convert_to_windows(trainD, model)
        testD = convert_to_windows(testD, model)
        
        print(f"  Training windows: {trainD.shape}")
        print(f"  Testing windows: {testD.shape}")
    except Exception as e:
        print(f"❌ Error converting to windows: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Get predictions
    print(f"\n🔮 Running model inference...")
    try:
        with torch.no_grad():
            lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
            loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"  Training loss shape: {lossT.shape}")
    print(f"  Test loss shape: {loss.shape}")
    print(f"  Predictions shape: {y_pred.shape}")
    
    # Align labels with predictions
    print(f"\n🔄 Aligning labels with predictions...")
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Handle different label formats based on dataset type
    if dataset_name in ['MBA', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'UCR', 'NAB', 'MSDS']:
        # Multivariate datasets - one-to-one label mapping
        if labels_np.shape[0] == loss.shape[0]:
            windowed_labels = labels_np
            print(f"  ✅ Labels already aligned (overlapping windows)")
        else:
            windowed_labels = labels_np
            print(f"  ⚠️  Shape mismatch - using labels as-is")
    elif labels_np.shape[0] != loss.shape[0]:
        # ECG and other datasets - need to aggregate/expand labels
        print(f"  Original labels: {labels_np.shape}")
        print(f"  Need to match: {loss.shape}")
        
        if len(labels_np.shape) == 2:
            if labels_np.shape[0] > loss.shape[0]:
                # Labels longer than predictions - aggregate (non-overlapping windows)
                original_samples = labels_np.shape[0]
                num_windows = loss.shape[0]
                window_size = original_samples // num_windows
                
                windowed_labels = []
                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = min((i + 1) * window_size, original_samples)
                    window_labels = labels_np[start_idx:end_idx, :]
                    aggregated = np.max(window_labels, axis=0, keepdims=True)
                    windowed_labels.append(aggregated)
                windowed_labels = np.vstack(windowed_labels)
                print(f"  ✓ Aggregated labels: {original_samples} → {len(windowed_labels)}")
            elif labels_np.shape[0] < loss.shape[0]:
                # Labels shorter - expand (repeat for windows)
                original_samples = labels_np.shape[0]
                windows_per_sample = loss.shape[0] // original_samples
                windowed_labels = np.repeat(labels_np, windows_per_sample, axis=0)
                windowed_labels = windowed_labels[:loss.shape[0]]
                print(f"  ✓ Expanded labels: {original_samples} → {len(windowed_labels)}")
            else:
                windowed_labels = labels_np
        else:
            windowed_labels = labels_np
    else:
        windowed_labels = labels_np
        print(f"  ✅ Labels match predictions")
    
    print(f"  Final shapes - loss: {loss.shape}, labels: {windowed_labels.shape}")
    
    # Compute evaluation metrics
    print(f"\n📊 Computing evaluation metrics...")
    df = pd.DataFrame()
    
    # Per-feature evaluation
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], windowed_labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    # Overall evaluation
    lossTfinal = np.mean(lossT, axis=1)
    lossFinal = np.mean(loss, axis=1)
    
    if len(windowed_labels.shape) > 1:
        labelsFinal = np.mean(windowed_labels, axis=1)
        labelsFinal = (labelsFinal >= 0.5).astype(int)
    else:
        labelsFinal = (np.sum(windowed_labels, axis=1) >= 1).astype(int)
    
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, windowed_labels))
    result.update(ndcg(loss, windowed_labels))
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 Test Results - {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n🎯 Overall Performance:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print(f"\n📈 Per-Feature Performance:")
    print(df.to_string(index=False))
    
    # Key metrics summary
    print(f"\n{'='*60}")
    print(f"🎉 Summary")
    print(f"{'='*60}")
    print(f"F1 Score:       {result.get('f1', 0):.6f}")
    print(f"Precision:      {result.get('precision', 0):.6f}")
    print(f"Recall:         {result.get('recall', 0):.6f}")
    
    # Extract FP and FN from precision/recall
    total_anomalies = np.sum(labelsFinal)
    total_normal = len(labelsFinal) - total_anomalies
    
    # From confusion matrix calculations
    tp = result.get('TP', 0)
    fp = result.get('FP', 0)
    fn = result.get('FN', 0)
    tn = result.get('TN', 0)
    
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")
    print(f"\nTotal anomalies in test set: {total_anomalies:.0f}")
    print(f"Total normal in test set: {total_normal:.0f}")
    
    # Performance assessment
    print(f"\n{'='*60}")
    if dataset_name == 'ecg_data':
        print(f"📌 Expected Performance (from main branch):")
        print(f"   F1: ~0.9999, FP: ~30, FN: ~1")
        if fp <= 50 and fn <= 5:
            print(f"   ✅ PASS - Performance meets expectations!")
        else:
            print(f"   ⚠️  REVIEW - Performance may need tuning")
    elif dataset_name == 'MBA':
        print(f"📌 Target Performance:")
        print(f"   F1: > 0.97")
        f1_score = result.get('f1', 0)
        if f1_score >= 0.97:
            print(f"   ✅ PASS - Achieved target F1 score!")
        else:
            print(f"   ⚠️  REVIEW - F1 below target, may need more training")
    
    print(f"{'='*60}")
    
    # Plot results
    print(f"\n📊 Generating visualization plots...")
    try:
        if isinstance(testO, torch.Tensor):
            testO_plot = testO.numpy()
        else:
            testO_plot = testO
        
        # Prepare data for plotting
        min_len = min(len(testO_plot), len(y_pred), len(loss), len(windowed_labels))
        testO_plot = testO_plot[:min_len]
        y_pred_plot = y_pred[:min_len]
        ascore_plot = loss[:min_len].reshape(-1, 1) if len(loss.shape) > 1 else loss[:min_len]
        labels_plot = windowed_labels[:min_len].reshape(-1, 1) if len(windowed_labels.shape) > 1 else windowed_labels[:min_len]
        
        testO_plot = np.roll(testO_plot, 1, 0)
        
        plotter(f'Optimized_Enhanced_DTAAD_{dataset_name}_TEST', testO_plot, y_pred_plot, ascore_plot, labels_plot)
        print(f"  ✅ Plots saved to output/Optimized_Enhanced_DTAAD_{dataset_name}_TEST/")
    except Exception as e:
        print(f"  ⚠️  Could not generate plots: {e}")
    
    return result

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Standalone Model Testing Script")
    print("   Optimized Enhanced DTAAD")
    print("="*60 + "\n")
    
    print("📝 Configuration:")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Dataset:    {DATASET}")
    print()
    
    # Allow command line override
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
        print(f"   (Dataset overridden from command line: {DATASET})")
    
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[2]
        print(f"   (Model path overridden from command line: {MODEL_PATH})")
    
    # Run test
    try:
        result = test_model(MODEL_PATH, DATASET)
        
        if result is not None:
            print(f"\n✅ Testing complete!\n")
        else:
            print(f"\n❌ Testing failed - see errors above\n")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
