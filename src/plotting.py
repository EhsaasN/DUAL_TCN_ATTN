import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

# Disable LaTeX and set figure size
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2
plt.ioff()  # Turn off interactive mode

os.makedirs('plots', exist_ok=True)


def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels):
    print(f"🔍 Plotter debug:")
    print(f"  y_true type: {type(y_true)}, shape: {y_true.shape if hasattr(y_true, 'shape') else 'No shape'}")
    print(f"  y_pred type: {type(y_pred)}, shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'No shape'}")
    print(f"  ascore type: {type(ascore)}, shape: {ascore.shape if hasattr(ascore, 'shape') else 'No shape'}")
    print(f"  labels type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else 'No shape'}")
    
    # FIX: Handle TranAD/DTAAD rolling BEFORE tensor conversion
    if 'TranAD' in name or 'DTAAD' in name: 
        if isinstance(y_true, torch.Tensor):
            y_true = torch.roll(y_true, 1, 0)
        else:
            y_true = np.roll(y_true, 1, 0)
    
    # FIX: Proper tensor to numpy conversion with device handling
    def safe_tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            # Move to CPU first if on GPU, then detach and convert
            if tensor.is_cuda:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.detach().numpy()
        else:
            return np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor
    
    # Convert all inputs safely
    y_true = safe_tensor_to_numpy(y_true)
    y_pred = safe_tensor_to_numpy(y_pred)
    ascore = safe_tensor_to_numpy(ascore)
    labels = safe_tensor_to_numpy(labels)
    
    # Ensure all arrays have at least 2 dimensions
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    if len(ascore.shape) == 1:
        ascore = ascore.reshape(-1, 1)
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    
    print(f"🎯 After preprocessing:")
    print(f"  y_true: {y_true.shape}, y_pred: {y_pred.shape}, ascore: {ascore.shape}, labels: {labels.shape}")
    
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    
    for dim in range(y_true.shape[1]):
        try:
            y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
            
            # Create figure with explicit cleanup
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
            ax1.set_ylabel('Value')
            ax1.set_title(f'Dimension = {dim}')
            
            ax1.plot(smooth(y_t), linewidth=0.2, label='True')
            ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
            ax3 = ax1.twinx()
            ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3, label='True Anomaly')
            if dim == 0: 
                ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
            
            ax2.plot(smooth(a_s), linewidth=0.2, color='g', label='Score')
            ax4 = ax2.twinx()
            ax4.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.3, label='Predicted Anomaly')
            if dim == 0: 
                ax4.legend(bbox_to_anchor=(1, 1.02))
            
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Anomaly Score')
            ax1.set_yticks([])
            ax2.set_yticks([])
            
            # Save and cleanup
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)  # Important: close figure to free memory
            
        except Exception as e:
            print(f"⚠️ Error plotting dimension {dim}: {e}")
            continue
    
    pdf.close()
    print(f"✅ Plots saved to: plots/{name}/output.pdf")