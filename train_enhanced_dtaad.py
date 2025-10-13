import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
from pprint import pprint

# Import your existing modules
from src.enhanced_dtaad import EnhancedDTAAD
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from main import convert_to_windows, load_dataset, backprop, save_model

def train_enhanced_dtaad():
    """Train the Enhanced DTAAD model"""
    print("🚀 Training Enhanced DTAAD for ECG Anomaly Detection")
    print("📊 Novel Architecture for Research Publication")
    print("=" * 60)
    
    # Load data using your existing function
    train_loader, test_loader, labels = load_dataset('ecg_data')
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    num_features = trainD.shape[1]
    
    print(f"📊 Dataset Information:")
    print(f"  Training samples: {trainD.shape}")
    print(f"  Testing samples: {testD.shape}")
    print(f"  Number of features: {num_features}")
    print(f"  Labels shape: {labels.shape}")
    
    # Create Enhanced DTAAD model
    model = EnhancedDTAAD(num_features).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    
    print(f"\n🏗️  Enhanced Model Architecture:")
    print(f"  ✅ Base DTAAD Framework")
    print(f"  ✅ ECG Cardiac Attention (Multi-head)")
    print(f"  ✅ Multi-Scale Temporal Extraction")
    print(f"  ✅ Enhanced Attention Mechanism (8 heads)")
    print(f"  ✅ Adaptive Fusion Weights")
    print(f"  ✅ Confidence Estimation")
    
    # Prepare data using your existing function
    trainO, testO = trainD, testD
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    print(f"\n🔄 Windowed Data:")
    print(f"  Training windows: {trainD.shape}")
    print(f"  Testing windows: {testD.shape}")
    
    # Training parameters
    num_epochs = 20  # Reasonable number without early stopping
    accuracy_list = []
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Training loop
    print(f"\n🏋️  Starting Training (40 epochs)...")
    start_time = time()
    
    for epoch in tqdm(range(num_epochs), desc="Training Enhanced DTAAD"):
        lossT, lr = backprop(epoch, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {lossT:.6f}, LR: {lr:.6f}")
    
    training_time = time() - start_time
    print(f"\n⏱️  Training completed in {training_time:.2f} seconds")
    
    # Save Enhanced model in same location as DTAAD but with different name
    save_path = 'checkpoints/Enhanced_DTAAD_ecg_data/'
    os.makedirs(save_path, exist_ok=True)
    
    # Save model using your existing save function format
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list
    }, f'{save_path}/model.ckpt')
    
    print(f"💾 Enhanced model saved to: {save_path}/model.ckpt")
    
    # Plot training curves
    plot_accuracies(accuracy_list, 'Enhanced_DTAAD_ecg_data')
    
    # Test the enhanced model
    print(f"\n🧪 Testing Enhanced DTAAD...")
    test_enhanced_model(model, trainD, testD, trainO, testO, labels, optimizer, scheduler)
    
    return model

def test_enhanced_model(model, trainD, testD, trainO, testO, labels, optimizer, scheduler):
    """Test the Enhanced DTAAD model using your existing evaluation framework"""
    
    # Testing phase
    torch.zero_grad = True
    model.eval()
    print(f"Testing Enhanced_DTAAD on ecg_data")
    
    # Get predictions using your existing backprop function
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    
    # Prepare data for plotting (using your existing logic)
    if isinstance(testO, torch.Tensor):
        testO_np = testO.detach().cpu().numpy()
    else:
        testO_np = testO
    
    # Take first sample and first channel for plotting
    if len(testO_np.shape) == 3:
        testO_plot = testO_np[0, 0, :]
    else:
        testO_plot = testO_np[0] if len(testO_np.shape) > 1 else testO_np
    
    # Prepare predictions for plotting
    y_pred_plot = y_pred[:len(testO_plot)] if len(y_pred) > len(testO_plot) else y_pred
    ascore_plot = loss[:len(testO_plot)] if len(loss) > len(testO_plot) else loss
    
    # Create labels for plotting
    labels_plot = np.zeros((len(testO_plot), 1))
    if len(labels.shape) > 1 and labels.shape[0] > 0:
        repeat_factor = len(testO_plot) // len(labels) + 1
        repeated_labels = np.tile(labels[:, 0], repeat_factor)
        labels_plot[:, 0] = repeated_labels[:len(testO_plot)]
    
    # Ensure all have same length
    min_len = min(len(testO_plot), len(y_pred_plot), len(ascore_plot))
    testO_plot = testO_plot[:min_len].reshape(-1, 1)
    y_pred_plot = y_pred_plot[:min_len].reshape(-1, 1)
    ascore_plot = ascore_plot[:min_len].reshape(-1, 1)
    labels_plot = labels_plot[:min_len].reshape(-1, 1)
    
    # Apply TranAD/DTAAD specific adjustment
    testO_plot = np.roll(testO_plot, 1, 0)
    
    # Plot results
    plotter('Enhanced_DTAAD_ecg_data', testO_plot, y_pred_plot, ascore_plot, labels_plot)
    
    # Evaluation using your existing framework
    df = pd.DataFrame()
    preds = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    
    # Create windowed labels to match windowed predictions
    if len(labels.shape) == 2 and labels.shape[0] < loss.shape[0]:
        original_samples = labels.shape[0]
        windows_per_sample = loss.shape[0] // original_samples
        windowed_labels = np.repeat(labels, windows_per_sample, axis=0)
        windowed_labels = windowed_labels[:loss.shape[0]]
    else:
        windowed_labels = labels

    # Evaluation per dimension
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], windowed_labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    # Final evaluation
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    if len(windowed_labels.shape) > 1:
        labelsFinal = np.mean(windowed_labels, axis=1)
        labelsFinal = (labelsFinal >= 0.5).astype(int)
    else:
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, windowed_labels))
    result.update(ndcg(loss, windowed_labels))
    
    print("\n📊 Enhanced DTAAD Results:")
    print(df)
    print("\n🎯 Overall Performance:")
    pprint(result)
    
    return result

if __name__ == "__main__":
    print("🎯 Enhanced DTAAD Training Script")
    print("👤 User: EhsaasN")
    print("📅 Date: 2025-10-11")
    print("🔬 Research: ECG Anomaly Detection with Enhanced Dual TCN Attention")
    
    enhanced_model = train_enhanced_dtaad()
    print("\n✅ Enhanced DTAAD training completed successfully!")
    print("📊 Model saved and ready for research publication!")