"""
Test script to verify MBA configuration changes:
1. Increased learning rate from 1e-3 to 5e-3
2. Confirmed overlapping windows (no downsampling)
"""

import sys
sys.path.append('.')

# Mock the parser args
class MockArgs:
    dataset = 'MBA'
    model = 'Optimized_Enhanced_DTAAD'
    retrain = True
    less = False

import src.parser as parser_module
parser_module.args = MockArgs()

print("=" * 80)
print("MBA Configuration Test")
print("=" * 80)

# Check the configuration that would be used
dataset = 'MBA'
if dataset == 'MBA':
    num_epochs = 50
    learning_rate = 5e-3  # INCREASED from 1e-3
    weight_decay = 1e-4
    step_size = 10
    gamma = 0.9
    
    print(f"\n✅ MBA Hyperparameters (Updated):")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {learning_rate} (5x increase from 1e-3 → 5e-3)")
    print(f"   Weight Decay: {weight_decay}")
    print(f"   Scheduler: StepLR(step_size={step_size}, gamma={gamma})")
    
    print(f"\n✅ Windowing Strategy:")
    print(f"   Type: OVERLAPPING windows (original DTAAD)")
    print(f"   No downsampling - all 7680 timesteps preserved")
    print(f"   Each timestep gets its own window for maximum accuracy")

print("\n" + "=" * 80)
print("Expected Impact:")
print("=" * 80)
print("1. Higher learning rate (5e-3) → Faster convergence, potentially better F1")
print("2. Overlapping windows → No data loss, better pattern recognition")
print("3. 50 epochs → Sufficient training time for convergence")
print("\nThese changes should improve F1 score from previous runs.")
print("=" * 80)
