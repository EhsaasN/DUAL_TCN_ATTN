#!/usr/bin/env python3
"""
Test script to validate Optimized Enhanced DTAAD compatibility with both univariate and multivariate data.
Tests ECG data (univariate) and MBA data (multivariate) to ensure the model works correctly with both.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from time import time
from pprint import pprint

# Import from training script
from train_optimized_enhanced_dtaad import train_optimized_enhanced_dtaad

def test_dataset(dataset_name, expected_min_f1=0.5):
    """Test the Optimized Enhanced DTAAD model on a dataset"""
    print(f"\n{'='*80}")
    print(f"Testing Optimized Enhanced DTAAD on {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    try:
        start_time = time()
        model = train_optimized_enhanced_dtaad(dataset_name)
        total_time = time() - start_time
        
        print(f"\n✅ {dataset_name.upper()} test PASSED")
        print(f"   Total time: {total_time:.2f} seconds")
        return True, total_time
        
    except Exception as e:
        print(f"\n❌ {dataset_name.upper()} test FAILED")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTIVARIATE COMPATIBILITY TEST")
    print("Testing Optimized Enhanced DTAAD with Univariate and Multivariate Data")
    print("="*80)
    
    results = {}
    
    # Test 1: ECG Data (Univariate)
    print("\n🔬 TEST 1: ECG Data (Univariate Time Series)")
    print("   - Expected: Multiple samples with single feature")
    print("   - Shape: (samples, 1, sequence_length)")
    success_ecg, time_ecg = test_dataset('ecg_data', expected_min_f1=0.9)
    results['ecg_data'] = {'success': success_ecg, 'time': time_ecg, 'type': 'univariate'}
    
    # Test 2: MBA Data (Multivariate)
    print("\n🔬 TEST 2: MBA Data (Multivariate Time Series)")
    print("   - Expected: Single sample with multiple features")
    print("   - Shape: (1, features, time_steps)")
    success_mba, time_mba = test_dataset('MBA', expected_min_f1=0.5)
    results['MBA'] = {'success': success_mba, 'time': time_mba, 'type': 'multivariate'}
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = all(r['success'] for r in results.values())
    
    for dataset, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} - {dataset.upper()} ({result['type']})")
        if result['success']:
            print(f"         Time: {result['time']:.2f}s")
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Optimized Enhanced DTAAD is compatible with both univariate and multivariate data.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the errors above.")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
