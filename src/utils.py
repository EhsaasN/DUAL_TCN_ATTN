import matplotlib.pyplot as plt
import os
import seaborn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
import torch



class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.png')
    plt.clf()


def plot_attention(model, layers, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    for layer in range(layers):  # layers
        fig, (axs, axs1) = plt.subplots(1, 2, figsize=(10, 4))
        heatmap = seaborn.heatmap(model.transformer_encoder1.layers[layer].att[0].data.cpu(), ax=axs)
        heatmap.set_title("Local_attention", fontsize=10)
        heatmap = seaborn.heatmap(model.transformer_encoder2.layers[layer].att[0].data.cpu(), ax=axs1)
        heatmap.set_title("Global_attention", fontsize=10)
    heatmap.get_figure().savefig(f'plots/{folder}/attention-score.png')
    plt.clf()


def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def getresults2(df, result):  # all dims-sum & mean
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall', 'ROC/AUC']:
        results2[a] = df2[a]
    results2['f1_mean'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2


def calculate_f1_precision_recall(y_true, y_pred, average='binary'):
    """
    Calculate F1 score, precision, and recall metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        Dictionary with f1, precision, recall scores
    """
    try:
        # Handle numpy arrays and tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Flatten arrays if needed
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Convert to binary if needed
        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred = (y_pred > 0.5).astype(int)
        
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        print(f'{color.RED}Error calculating metrics: {e}{color.ENDC}')
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}


def calculate_precision_recall_curve(y_true, y_scores):
    """
    Calculate precision-recall curve and area under curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores (probabilities)
    
    Returns:
        Dictionary with precision, recall arrays and AUC-PR score
    """
    try:
        # Handle numpy arrays and tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.cpu().numpy()
        
        # Flatten arrays
        y_true = np.asarray(y_true).flatten()
        y_scores = np.asarray(y_scores).flatten()
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'pr_auc': pr_auc
        }
    except Exception as e:
        print(f'{color.RED}Error calculating PR curve: {e}{color.ENDC}')
        return {'precision': [], 'recall': [], 'thresholds': [], 'pr_auc': 0.0}


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    def __init__(self, patience=7, min_delta=0, verbose=True, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
            mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            score_improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            score_improved = score > (self.best_score + self.min_delta)
        
        if score_improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'{color.GREEN}Validation improved to {score:.6f}{color.ENDC}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'{color.RED}EarlyStopping counter: {self.counter}/{self.patience}{color.ENDC}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'{color.BOLD}Early stopping triggered. Best score: {self.best_score:.6f} at epoch {self.best_epoch}{color.ENDC}')
                return True
        
        return False
