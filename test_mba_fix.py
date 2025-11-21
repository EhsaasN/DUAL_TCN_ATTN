"""
Test to verify MBA dataset loading fix
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create mock MBA data
os.makedirs('processed/MBA', exist_ok=True)
train_data = np.random.rand(7680, 2)
test_data = np.random.rand(7680, 2)
labels_data = np.random.randint(0, 2, (7680, 2))

np.save('processed/MBA/train.npy', train_data)
np.save('processed/MBA/test.npy', test_data)
np.save('processed/MBA/labels.npy', labels_data)

# Mock args
class MockArgs:
    dataset = 'MBA'
    model = 'Optimized_Enhanced_DTAAD'
    retrain = True
    less = False

import src.parser as parser_module
parser_module.args = MockArgs()

from main import load_dataset

print("=" * 80)
print("Testing MBA Dataset Loading Fix")
print("=" * 80)

train_loader, test_loader, labels = load_dataset('MBA')

import torch
trainD = next(iter(train_loader))
testD = next(iter(test_loader))

print(f"\n✅ Results:")
print(f"   Train batch shape: {trainD.shape}")
print(f"   Test batch shape: {testD.shape}")  
print(f"   Labels shape: {labels.shape}")

print(f"\n📊 Expected:")
print(f"   Train: (7680, 2) - full dataset as one batch")
print(f"   Test: (7680, 2) - full dataset as one batch")
print(f"   Labels: (7680, 2)")

if trainD.shape == torch.Size([7680, 2]) and testD.shape == torch.Size([7680, 2]):
    print(f"\n✅ SUCCESS: Data shapes are correct!")
else:
    print(f"\n❌ FAILED: Data shapes are incorrect!")
    
print("=" * 80)
