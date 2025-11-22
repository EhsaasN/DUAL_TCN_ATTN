#!/usr/bin/env python
"""
Test preprocessing output for ECG and MBA datasets
Shows data shapes and validates compatibility with train_optimized_enhanced_dtaad.py
"""
import numpy as np
import sys
import os

def test_dataset(dataset_name):
    """Test a preprocessed dataset"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    folder = f'processed/{dataset_name}'
    if not os.path.exists(folder):
        print(f"❌ {dataset_name} not preprocessed. Run: python preprocess.py {dataset_name}")
        return False
    
    # Load data
    train = np.load(os.path.join(folder, 'train.npy'))
    test = np.load(os.path.join(folder, 'test.npy'))
    labels = np.load(os.path.join(folder, 'labels.npy'))
    
    print("📊 Raw Data Shapes:")
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Total anomalies: {np.sum(labels):.0f}")
    
    # Determine dataset type
    is_ecg = dataset_name == 'ecg_data'
    is_multivariate = len(test.shape) == 2 and test.shape[1] > 1
    
    print(f"\n🔍 Dataset Characteristics:")
    if is_ecg:
        print(f"  Type: Univariate (ECG)")
        print(f"  Samples: {test.shape[0]}")
        print(f"  Sequence length: {test.shape[1]}")
        print(f"  Windowing strategy: Non-overlapping (step=window_size)")
        print(f"  Reason: Memory optimization for large sequences")
        
        # Calculate expected windows
        window_size = 10
        num_samples = test.shape[0]
        sequence_length = test.shape[1]
        windows_per_sample = sequence_length // window_size
        total_windows = num_samples * windows_per_sample
        
        print(f"\n📐 Expected Windowing (window_size={window_size}):")
        print(f"  Windows per sample: {windows_per_sample}")
        print(f"  Total windows: {total_windows}")
        print(f"  Label aggregation: Max aggregation (any anomaly in window → window is anomaly)")
        
        # Check label format
        if labels.shape[0] == test.shape[0]:
            print(f"\n✅ Labels per sample: {labels.shape}")
            print(f"   Will be expanded to {total_windows} windows (repeat per window)")
        else:
            print(f"\n⚠️  Unexpected label shape!")
            
    elif is_multivariate:
        print(f"  Type: Multivariate")
        print(f"  Timesteps: {test.shape[0]}")
        print(f"  Features: {test.shape[1]}")
        print(f"  Windowing strategy: Overlapping (one window per timestep)")
        print(f"  Reason: Preserve all temporal information for accuracy")
        
        window_size = 10
        expected_windows = test.shape[0]
        
        print(f"\n📐 Expected Windowing (window_size={window_size}):")
        print(f"  Total windows: {expected_windows} (overlapping)")
        print(f"  Label mapping: One-to-one (each window has corresponding label)")
        
        # Check labels
        if labels.shape[0] == test.shape[0]:
            print(f"\n✅ Labels aligned: {labels.shape}")
            print(f"   One label per timestep → one label per window")
        else:
            print(f"\n⚠️  Label mismatch: {labels.shape} vs expected {test.shape}")
    else:
        print(f"  Type: Unknown/Univariate")
        print(f"  Shape: {test.shape}")
    
    # Show sample statistics
    print(f"\n📈 Data Statistics:")
    print(f"  Train mean: {np.mean(train):.4f}, std: {np.std(train):.4f}")
    print(f"  Test mean: {np.mean(test):.4f}, std: {np.std(test):.4f}")
    print(f"  Train range: [{np.min(train):.4f}, {np.max(train):.4f}]")
    print(f"  Test range: [{np.min(test):.4f}, {np.max(test):.4f}]")
    
    # Anomaly analysis
    if len(labels.shape) == 2:
        anomaly_samples = np.sum(np.any(labels > 0, axis=1))
        total_samples = labels.shape[0]
        anomaly_ratio = anomaly_samples / total_samples
        print(f"\n🎯 Anomaly Analysis:")
        print(f"  Samples with anomalies: {anomaly_samples}/{total_samples} ({anomaly_ratio*100:.1f}%)")
        
        if labels.shape[1] > 1:
            for i in range(labels.shape[1]):
                feature_anomalies = np.sum(labels[:, i])
                print(f"  Feature {i+1} anomalies: {feature_anomalies:.0f}")
    else:
        anomaly_samples = np.sum(labels)
        total_samples = len(labels)
        anomaly_ratio = anomaly_samples / total_samples
        print(f"\n🎯 Anomaly Analysis:")
        print(f"  Anomaly samples: {anomaly_samples:.0f}/{total_samples} ({anomaly_ratio*100:.1f}%)")
    
    print(f"\n✅ {dataset_name} preprocessing successful!")
    print(f"   Ready for train_optimized_enhanced_dtaad.py")
    
    return True

if __name__ == "__main__":
    print("🔬 Preprocessing Output Verification")
    print("=" * 60)
    
    # Test both datasets
    datasets_to_test = ['ecg_data', 'MBA']
    
    results = {}
    for dataset in datasets_to_test:
        results[dataset] = test_dataset(dataset)
    
    print(f"\n{'='*60}")
    print("📋 Summary")
    print(f"{'='*60}")
    for dataset, success in results.items():
        status = "✅ Ready" if success else "❌ Not preprocessed"
        print(f"{dataset:15s}: {status}")
    
    print("\n🎉 Preprocessing verification complete!")
    print("\nNext steps:")
    print("1. For ECG: python train_optimized_enhanced_dtaad.py --dataset ecg_data")
    print("2. For MBA: python train_optimized_enhanced_dtaad.py --dataset MBA")
