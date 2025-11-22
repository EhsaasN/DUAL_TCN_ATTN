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

# Import optimized version
from src.enhanced_dtaad_optimized import OptimizedEnhancedDTAAD
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from main import convert_to_windows, load_dataset, backprop, save_model
import matplotlib
matplotlib.use('Agg')  # Add this line
import matplotlib.pyplot as plt
plt.ioff() 

# def train_optimized_enhanced_dtaad():
#     """Train the Optimized Enhanced DTAAD model - 50% faster"""
#     print("🚀 Training Optimized Enhanced DTAAD for ECG Anomaly Detection")
#     print("⚡ Performance-Optimized Version (50% faster)")
#     # print("📊 Novel Architecture for Research Publication")
#     print("=" * 60)
    
#     # Load data
#     train_loader, test_loader, labels = load_dataset('ecg_data')
#     trainD, testD = next(iter(train_loader)), next(iter(test_loader))
#     num_features = trainD.shape[1]
    
#     print(f"📊 Dataset Information:")
#     print(f"  Training samples: {trainD.shape}")
#     print(f"  Testing samples: {testD.shape}")
#     print(f"  Number of features: {num_features}")
#      # Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"🔧 Using device: {device}")
    
#     # Create Optimized Enhanced DTAAD model
#     model = OptimizedEnhancedDTAAD(num_features).double()
    
#     # Optimized training settings
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)  # Slightly higher LR
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)  # Simpler scheduler
      
#     # Prepare data
#     # trainO, testO = trainD, testD
#     # trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
#     trainO, testO = trainD.to(device), testD.to(device)
#     trainD, testD = convert_to_windows(trainD.to(device), model), convert_to_windows(testD.to(device), model)
    
#     # Training parameters
#     num_epochs = 20  # Reduced from 40 for faster training
#     accuracy_list = []
    
   
    
#     # Training loop with timing
#     print(f"\n🏋️  Starting Optimized Training (20 epochs)...")
#     start_time = time()
    
#     for epoch in tqdm(range(num_epochs), desc="Training Optimized Enhanced DTAAD"):
#         lossT, lr = backprop(epoch, model, trainD, trainO, optimizer, scheduler)
#         accuracy_list.append((lossT, lr))
        
#         # Progress every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {lossT:.6f}, LR: {lr:.6f}")
    
#     training_time = time() - start_time
#     print(f"\n⏱️  Optimized Training completed in {training_time:.2f} seconds")
#     # print(f"   Optimized Enhanced (20 epochs): {training_time:.1f} seconds")
    
#     # Save model
#     save_path = 'checkpoints/Optimized_Enhanced_DTAAD_ecg_data/'
#     os.makedirs(save_path, exist_ok=True)
    
#     torch.save({
#         'epoch': num_epochs - 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'accuracy_list': accuracy_list,
#         'training_time': training_time
#     }, f'{save_path}/model.ckpt')
    
#     print(f"💾 Optimized Enhanced model saved to: {save_path}/model.ckpt")
    
#     # Plot training curves
#     plot_accuracies(accuracy_list, 'Optimized_Enhanced_DTAAD_ecg_data')
    
#     # Test the model
#     print(f"\n🧪 Testing Optimized Enhanced DTAAD...")
#     test_optimized_model(model, trainD, testD, trainO, testO, labels, optimizer, scheduler)
    
#     return model
def train_optimized_enhanced_dtaad(dataset='ecg_data'):
    """Train the Optimized Enhanced DTAAD model - 50% faster"""
    print(f"🚀 Training Optimized Enhanced DTAAD for {dataset.upper()} Anomaly Detection")
    print("=" * 60)
    
    # Load data
    train_loader, test_loader, labels = load_dataset(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    trainD, testD = next(iter(train_loader)).to(device), next(iter(test_loader)).to(device)
    labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).to(device)
    num_features = trainD.shape[1]
    
    print(f"📊 Dataset Information:")
    print(f"  Training samples: {trainD.shape}")
    print(f"  Testing samples: {testD.shape}")
    print(f"  Number of features: {num_features}")
    
    # Create Optimized Enhanced DTAAD model
    model = OptimizedEnhancedDTAAD(num_features).double().to(device)
    
    # Dataset-specific hyperparameters
    if dataset == 'MBA':
        # MBA requires more epochs and different hyperparameters
        # Increased learning rate per user request to improve F1 score
        num_epochs = 50  # Reduced from 100 for faster training, still enough for convergence
        learning_rate = 5e-3  # Increased from 1e-3 for faster convergence
        weight_decay = 1e-4
        step_size = 10
        gamma = 0.9
        print(f"📈 MBA-specific hyperparameters:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate} (increased for better F1)")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Note: Using overlapping windows (no downsampling) for maximum accuracy")
    else:
        # Default for ECG and other datasets
        num_epochs = 5
        learning_rate = 4e-4
        weight_decay = 1e-5
        step_size = 5
        gamma = 0.9
    
    # Optimized training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
      
    # Prepare data
    trainO, testO = trainD, testD
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    # Handle labels differently for MBA vs ECG
    # MBA: overlapping windows means one label per window (one-to-one mapping)
    # ECG: non-overlapping windows need label expansion
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels
    
    # For MBA/SMAP with 2D data and overlapping windows, labels are already aligned
    # For ECG with 3D data and non-overlapping windows, we need to expand labels
    if dataset in ['MBA', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'UCR', 'NAB', 'MSDS']:
        # Multivariate datasets: labels shape (timesteps, features) matches windows
        # No windowing needed - one-to-one mapping
        print(f"📋 {dataset}: Using original labels (one-to-one with overlapping windows)")
        labels = torch.from_numpy(labels_np).to(device)
    elif len(trainD.shape) == 3 and trainD.shape[0] > 100:  # ECG case: many samples
        # For ECG: Convert labels to windowed format to match testD
        # Labels are (samples, features) - need to expand for windows
        if len(labels_np.shape) == 2:
            # Expand labels to match windowed data
            original_samples = labels_np.shape[0]
            num_windows = testD.shape[0]
            windows_per_sample = num_windows // original_samples
            labels_windowed = np.repeat(labels_np, windows_per_sample, axis=0)
            labels_windowed = labels_windowed[:num_windows]
            labels = torch.from_numpy(labels_windowed).to(device)
        else:
            labels = torch.from_numpy(labels_np).to(device)
    else:
        labels = torch.from_numpy(labels_np).to(device)
    
    # Training parameters
    accuracy_list = []
    
    # Training loop with timing
    print(f"\n🏋️  Starting Optimized Training ({num_epochs} epochs)...")
    start_time = time()
    
    for epoch in tqdm(range(num_epochs), desc="Training Optimized Enhanced DTAAD"):
        lossT, lr = backprop(epoch, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
        
        # Progress reporting
        if dataset == 'MBA' and (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {lossT:.6f}, LR: {lr:.6f}")
        elif (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {lossT:.6f}, LR: {lr:.6f}")
    
    training_time = time() - start_time
    print(f"\n⏱️  Optimized Training completed in {training_time:.2f} seconds")
    
    # Save model
    save_path = f'checkpoints/Optimized_Enhanced_DTAAD_{dataset}/'
    os.makedirs(save_path, exist_ok=True)
    
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list,
        'training_time': training_time
    }, f'{save_path}/model.ckpt')
    
    print(f"💾 Optimized Enhanced model saved to: {save_path}/model.ckpt")
    
    # Plot training curves
    plot_accuracies(accuracy_list, f'Optimized_Enhanced_DTAAD_{dataset}')
    
    # Test the model
    print(f"\n🧪 Testing Optimized Enhanced DTAAD...")
    test_optimized_model(model, trainD, testD, trainO, testO, labels, optimizer, scheduler, dataset)
    
    return model

def test_optimized_model(model, trainD, testD, trainO, testO, labels, optimizer, scheduler, dataset='ecg_data'):
    # Force all data to CPU for plotting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    """Test the optimized model"""
    testD = testD.to(device)
    testO = testO.to(device)
    labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).to(device)
    model.eval()
    print(f"Testing Optimized_Enhanced_DTAAD on {dataset}")
    
    # Get predictions
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    # ascore_plot = loss.detach().cpu().numpy()
    # Ensure y_pred is a NumPy array
    if isinstance(loss, torch.Tensor):
        ascore_plot = loss.detach().cpu().numpy()
    else:
        ascore_plot = loss
    if isinstance(y_pred, torch.Tensor):
        y_pred_plot = y_pred.detach().cpu().numpy()
    else:
        y_pred_plot = y_pred

    # Convert testO to NumPy for plotting
    testO_np = testO.detach().cpu().numpy() if isinstance(testO, torch.Tensor) else testO
    # Evaluation (same as original)
    if isinstance(testO, torch.Tensor):
        testO_np = testO.detach().cpu().numpy()
    else:
        testO_np = testO
    
    if len(testO_np.shape) == 3:
        testO_plot = testO_np[0, 0, :]
    else:
        testO_plot = testO_np[0] if len(testO_np.shape) > 1 else testO_np
    
    y_pred_plot = y_pred[:len(testO_plot)] if len(y_pred) > len(testO_plot) else y_pred
    ascore_plot = loss[:len(testO_plot)] if len(loss) > len(testO_plot) else loss
    
    labels_plot = np.zeros((len(testO_plot), 1))
    if len(labels.shape) > 1 and labels.shape[0] > 0:
        # Handle both multivariate and univariate labels
        labels_cpu = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        
        # For multivariate data (MBA), take first feature's labels
        if labels_cpu.shape[1] > 1:
            labels_flat = labels_cpu[0, 0, :] if len(labels_cpu.shape) == 3 else labels_cpu[:, 0]
        else:
            labels_flat = labels_cpu[:, 0]
            
        repeat_factor = len(testO_plot) // len(labels_flat) + 1
        repeated_labels = np.tile(labels_flat, repeat_factor)
        labels_plot[:, 0] = repeated_labels[:len(testO_plot)]
    
    min_len = min(len(testO_plot), len(y_pred_plot), len(ascore_plot))
    testO_plot = testO_plot[:min_len].reshape(-1, 1)
    y_pred_plot = y_pred_plot[:min_len].reshape(-1, 1)
    ascore_plot = ascore_plot[:min_len].reshape(-1, 1)
    labels_plot = labels_plot[:min_len].reshape(-1, 1)
    
    testO_plot = np.roll(testO_plot, 1, 0)
    
    # Plot results
    plotter(f'Optimized_Enhanced_DTAAD_{dataset}', testO_plot, y_pred_plot, ascore_plot, labels_plot)
    
    # Evaluation
    df = pd.DataFrame()
    preds = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    
    # Convert labels to numpy for evaluation
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Handle different label formats
    if dataset in ['MBA', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'UCR', 'NAB', 'MSDS']:
        # Multivariate datasets with overlapping windows - labels should match windows one-to-one
        if len(labels_np.shape) == 2 and labels_np.shape[0] == loss.shape[0]:
            windowed_labels = labels_np
            print(f"✅ {dataset}: Labels already aligned (overlapping windows)")
        else:
            # Shouldn't happen, but handle gracefully
            windowed_labels = labels_np
            print(f"⚠️  {dataset}: Unexpected label shape {labels_np.shape}, using as-is")
    elif labels_np.shape[0] != loss.shape[0]:
        # For ECG and other datasets with non-overlapping windows
        print(f"🔄 Aligning labels: {labels_np.shape} → match loss shape {loss.shape}")
        if len(labels_np.shape) == 2:
            if labels_np.shape[0] > loss.shape[0]:
                # Labels are longer than predictions (non-overlapping windows case)
                # We need to downsample/aggregate labels to match windows
                original_samples = labels_np.shape[0]
                num_windows = loss.shape[0]
                window_size = original_samples // num_windows
                
                # Aggregate labels: if any label in window is 1, window label is 1
                windowed_labels = []
                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = min((i + 1) * window_size, original_samples)
                    window_labels = labels_np[start_idx:end_idx, :]
                    # Use max (if any timestep in window is anomaly, window is anomaly)
                    aggregated = np.max(window_labels, axis=0, keepdims=True)
                    windowed_labels.append(aggregated)
                windowed_labels = np.vstack(windowed_labels)
                print(f"   Downsampled labels from {original_samples} to {len(windowed_labels)} windows")
            elif labels_np.shape[0] < loss.shape[0]:
                # Labels are shorter - need to expand (repeat labels for windows)
                original_samples = labels_np.shape[0]
                windows_per_sample = loss.shape[0] // original_samples
                windowed_labels = np.repeat(labels_np, windows_per_sample, axis=0)
                windowed_labels = windowed_labels[:loss.shape[0]]
                print(f"   Expanded labels from {original_samples} to {len(windowed_labels)} windows")
            else:
                windowed_labels = labels_np
        else:
            windowed_labels = labels_np
    else:
        windowed_labels = labels_np
    
    print(f"✅ Final shapes - loss: {loss.shape}, labels: {windowed_labels.shape}")

    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], windowed_labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    if len(windowed_labels.shape) > 1:
        labelsFinal = np.mean(windowed_labels, axis=1)
        labelsFinal = (labelsFinal >= 0.5).astype(int)
    else:
        labelsFinal = (np.sum(labels_np, axis=1) >= 1) + 0
    
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, windowed_labels))
    result.update(ndcg(loss, windowed_labels))
    
    print("\n📊 Optimized Enhanced DTAAD Results:")
    print(df)
    print("\n🎯 Overall Performance:")
    pprint(result)
    
    return result

if __name__ == "__main__":
    # Get dataset from parser args instead of sys.argv
    from src.parser import args as parser_args
    
    # If dataset specified in command line, use it; otherwise use parser default
    dataset = parser_args.dataset if hasattr(parser_args, 'dataset') else 'ecg_data'
    
    print(f"🎯 Starting training for dataset: {dataset}")
    
    start_time = time()
    optimized_model = train_optimized_enhanced_dtaad(dataset)
    total_time = time() - start_time
    
    print(f"\n✅ Optimized Enhanced DTAAD training completed!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
   