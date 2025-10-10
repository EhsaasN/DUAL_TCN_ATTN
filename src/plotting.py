import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

# plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)


def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels):
    # print(f"🔍 Plotter debug:")
    # print(f"  y_true shape: {y_true.shape if hasattr(y_true, 'shape') else type(y_true)}")
    # print(f"  y_pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else type(y_pred)}")
    # print(f"  ascore shape: {ascore.shape if hasattr(ascore, 'shape') else type(ascore)}")  # FIX: changed 'loss' to 'ascore'
    # print(f"  labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
    
    # FIX: Correct condition logic
    if 'TranAD' in name or 'DTAAD' in name: 
        if isinstance(y_true, torch.Tensor):
            y_true = torch.roll(y_true, 1, 0)
        else:
            y_true = np.roll(y_true, 1, 0)
    
    # Convert tensors to numpy arrays for plotting
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(ascore, torch.Tensor):
        ascore = ascore.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
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
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_ylabel('Value')
            ax1.set_title(f'Dimension = {dim}')
            
            ax1.plot(smooth(y_t), linewidth=0.2, label='True')
            ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
            ax3 = ax1.twinx()
            ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3, label='True Anomaly')
            if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
            
            ax2.plot(smooth(a_s), linewidth=0.2, color='g', label='Score')
            ax4 = ax2.twinx()
            ax4.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.3, label='Predicted Anomaly')
            if dim == 0: ax4.legend(bbox_to_anchor=(1, 1.02))
            
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Anomaly Score')
            ax1.set_yticks([])
            ax2.set_yticks([])
            pdf.savefig(fig)
            plt.close()
        except Exception as e:
            print(f"⚠️ Error plotting dimension {dim}: {e}")
            continue
    
    pdf.close()
    print(f"📊 Plots saved to plots/{name}/output.pdf")