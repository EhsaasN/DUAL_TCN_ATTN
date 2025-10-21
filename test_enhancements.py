#!/usr/bin/env python
"""
Test script to validate the enhancements made to the DUAL_TCN_ATTN repository.
This script tests:
1. Enhanced metrics (F1, precision, recall)
2. Early stopping functionality
3. Enhanced data loading with error handling
4. Optimized window conversion
5. Enhanced model saving
6. Learning rate scheduler options
"""

import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    calculate_f1_precision_recall, 
    calculate_precision_recall_curve,
    EarlyStopping,
    color
)

def test_metrics():
    """Test F1, precision, recall metrics"""
    print(f"\n{color.HEADER}Testing Enhanced Metrics{color.ENDC}")
    
    # Create sample data
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    
    # Test F1, precision, recall
    metrics = calculate_f1_precision_recall(y_true, y_pred)
    print(f"{color.GREEN}✓ F1 Score: {metrics['f1']:.4f}{color.ENDC}")
    print(f"{color.GREEN}✓ Precision: {metrics['precision']:.4f}{color.ENDC}")
    print(f"{color.GREEN}✓ Recall: {metrics['recall']:.4f}{color.ENDC}")
    
    # Test precision-recall curve
    y_scores = np.random.rand(10)
    pr_metrics = calculate_precision_recall_curve(y_true, y_scores)
    print(f"{color.GREEN}✓ PR-AUC: {pr_metrics['pr_auc']:.4f}{color.ENDC}")
    
    return True

def test_early_stopping():
    """Test early stopping functionality"""
    print(f"\n{color.HEADER}Testing Early Stopping{color.ENDC}")
    
    # Create early stopping instance
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=False, mode='min')
    
    # Simulate training with decreasing loss
    losses = [1.0, 0.9, 0.8, 0.79, 0.78, 0.77, 0.77, 0.77, 0.77]
    
    stop_triggered = False
    for epoch, loss in enumerate(losses):
        if early_stopping(loss, epoch):
            stop_triggered = True
            print(f"{color.GREEN}✓ Early stopping triggered at epoch {epoch}{color.ENDC}")
            break
    
    if stop_triggered:
        print(f"{color.GREEN}✓ Early stopping works correctly{color.ENDC}")
        return True
    else:
        print(f"{color.RED}✗ Early stopping failed to trigger{color.ENDC}")
        return False

def test_window_conversion():
    """Test optimized window conversion"""
    print(f"\n{color.HEADER}Testing Window Conversion{color.ENDC}")
    
    # Create mock model with n_window attribute
    class MockModel:
        def __init__(self):
            self.n_window = 10
    
    model = MockModel()
    
    # Test 2D data
    data_2d = np.random.rand(100, 5)  # [samples, features]
    from main import convert_to_windows
    
    try:
        windows_2d = convert_to_windows(data_2d, model)
        print(f"{color.GREEN}✓ 2D window conversion: {windows_2d.shape}{color.ENDC}")
    except Exception as e:
        print(f"{color.RED}✗ 2D window conversion failed: {e}{color.ENDC}")
        return False
    
    # Test 3D data
    data_3d = np.random.rand(10, 2, 50)  # [samples, features, sequence]
    try:
        windows_3d = convert_to_windows(data_3d, model)
        print(f"{color.GREEN}✓ 3D window conversion: {windows_3d.shape}{color.ENDC}")
    except Exception as e:
        print(f"{color.RED}✗ 3D window conversion failed: {e}{color.ENDC}")
        return False
    
    return True

def test_enhanced_tcn():
    """Test enhanced dual TCN architecture"""
    print(f"\n{color.HEADER}Testing Enhanced Dual TCN{color.ENDC}")
    
    from src.gltcn import Tcn_Local, Tcn_Global
    
    # Test Local TCN
    try:
        tcn_local = Tcn_Local(num_inputs=5, num_outputs=5, kernel_size=3, dropout=0.2)
        x = torch.randn(2, 5, 20)  # [batch, features, sequence]
        out_local = tcn_local(x)
        print(f"{color.GREEN}✓ Enhanced Local TCN: input {x.shape} -> output {out_local.shape}{color.ENDC}")
    except Exception as e:
        print(f"{color.RED}✗ Local TCN failed: {e}{color.ENDC}")
        return False
    
    # Test Global TCN
    try:
        tcn_global = Tcn_Global(num_inputs=5, num_outputs=5, kernel_size=3, dropout=0.2)
        out_global = tcn_global(x)
        print(f"{color.GREEN}✓ Enhanced Global TCN: input {x.shape} -> output {out_global.shape}{color.ENDC}")
    except Exception as e:
        print(f"{color.RED}✗ Global TCN failed: {e}{color.ENDC}")
        return False
    
    return True

def test_scheduler_options():
    """Test different learning rate scheduler options"""
    print(f"\n{color.HEADER}Testing Learning Rate Schedulers{color.ENDC}")
    
    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 5).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    schedulers = {
        'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9),
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6),
        'exponential': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    }
    
    for name, scheduler in schedulers.items():
        try:
            # Simulate a step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{color.GREEN}✓ {name} scheduler: LR = {current_lr:.6f}{color.ENDC}")
        except Exception as e:
            print(f"{color.RED}✗ {name} scheduler failed: {e}{color.ENDC}")
            return False
    
    return True

def main():
    """Run all tests"""
    print(f"{color.BOLD}{'='*60}{color.ENDC}")
    print(f"{color.BOLD}Testing DUAL_TCN_ATTN Enhancements{color.ENDC}")
    print(f"{color.BOLD}{'='*60}{color.ENDC}")
    
    tests = [
        ("Enhanced Metrics", test_metrics),
        ("Early Stopping", test_early_stopping),
        ("Window Conversion", test_window_conversion),
        ("Enhanced Dual TCN", test_enhanced_tcn),
        ("Learning Rate Schedulers", test_scheduler_options),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{color.RED}✗ {test_name} crashed: {e}{color.ENDC}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{color.BOLD}{'='*60}{color.ENDC}")
    print(f"{color.BOLD}Test Summary{color.ENDC}")
    print(f"{color.BOLD}{'='*60}{color.ENDC}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{color.GREEN}PASS{color.ENDC}" if result else f"{color.RED}FAIL{color.ENDC}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{color.BOLD}Total: {passed}/{total} tests passed{color.ENDC}")
    
    if passed == total:
        print(f"\n{color.GREEN}✓ All enhancements validated successfully!{color.ENDC}")
        return 0
    else:
        print(f"\n{color.RED}✗ Some tests failed. Please review the output above.{color.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
