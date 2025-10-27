import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import importlib
import sys
if 'src.models' in sys.modules:
    importlib.reload(sys.modules['src.models'])

def convert_to_windows(data, model):
    """
    Optimized window conversion with better memory efficiency and validation.
    
    Args:
        data: Input data to convert to windows
        model: Model with n_window attribute
        
    Returns:
        Windowed data tensor
    """
    windows = []
    w_size = model.n_window
    
    # Validate window size
    if w_size <= 0:
        raise ValueError(f'{color.RED}Invalid window size: {w_size}{color.ENDC}')
    
    # Convert to tensor first with validation
    try:
        data = torch.tensor(data, dtype=torch.float64)
    except Exception as e:
        print(f'{color.RED}Error converting data to tensor: {e}{color.ENDC}')
        raise
    
    # Validate data is not empty
    if data.numel() == 0:
        raise ValueError(f'{color.RED}Empty data provided for windowing{color.ENDC}')
    
    # Handle 3D data [samples, features, sequence]
    if len(data.shape) == 3:
        samples, features, sequence_length = data.shape
        print(f"🔍 Processing 3D data: samples={samples}, features={features}, sequence_length={sequence_length}")
        
        # Validate window size vs sequence length
        if w_size > sequence_length:
            print(f'{color.RED}Warning: Window size ({w_size}) > sequence length ({sequence_length}). Using sequence_length as window size.{color.ENDC}')
            w_size = sequence_length
        
        # Use non-overlapping windows for efficiency (can be changed to w_size // 2 for 50% overlap)
        step_size = w_size
        
        # Pre-allocate list size for efficiency
        num_windows = (sequence_length - w_size) // step_size + 1
        windows = []
        
        # Create windows with step size
        for i in range(0, sequence_length - w_size + 1, step_size):
            w = data[:, :, i:i + w_size]  # [samples, features, window_size]
            windows.append(w)
        
        if windows:
            # Stack and reshape efficiently
            windows = torch.stack(windows, dim=0)  # [num_windows, samples, features, window_size]
            num_windows, samples, features, window_size = windows.shape
            windows = windows.view(num_windows * samples, features, window_size)
            print(f"🎯 Created {num_windows} windows per sample, total windows: {windows.shape[0]}")
        else:
            print(f'{color.RED}Warning: No windows created, returning original data{color.ENDC}')
            windows = data
            
    else:  # 2D data [samples, features] - original logic with optimizations
        num_features = data.shape[1]
        sequence_length = data.shape[0]
        
        # Validate window size
        if w_size > sequence_length:
            print(f'{color.RED}Warning: Window size ({w_size}) > data length ({sequence_length}). Using data_length as window size.{color.ENDC}')
            w_size = sequence_length
        
        step_size = w_size  # Non-overlapping
        
        # Pre-allocate for efficiency
        num_windows = (sequence_length - w_size) // step_size + 1
        windows = []
        
        for i in range(0, sequence_length - w_size + 1, step_size):
            w = data[i:i + w_size]  # [window_size, num_features]
            windows.append(w)
        
        if windows:
            windows = torch.stack(windows)  # [num_windows, window_size, num_features]
            windows = windows.permute(0, 2, 1)  # [num_windows, num_features, window_size]
            print(f"🎯 Created {len(windows)} windows from 2D data")
        else:
            print(f'{color.RED}Warning: No windows created, returning reshaped data{color.ENDC}')
            windows = data.unsqueeze(0).permute(0, 2, 1)
    
    # Final validation
    if windows.numel() == 0:
        raise ValueError(f'{color.RED}Window conversion resulted in empty tensor{color.ENDC}')
    
    return windows


def load_dataset(dataset):
    """
    Enhanced data loading with error handling and validation.
    
    Args:
        dataset: Dataset name to load
        
    Returns:
        train_loader, test_loader, labels
        
    Raises:
        Exception: If data validation fails
    """
    folder = os.path.join(output_folder, dataset)
    
    # Enhanced error handling
    if not os.path.exists(folder):
        raise Exception(f'{color.RED}Processed data folder not found: {folder}{color.ENDC}')
    
    loader = []
    required_files = ['train', 'test', 'labels']
    
    for file in required_files:
        file_path = os.path.join(folder, f'{file}.npy')
        
        try:
            # Check file existence
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'{color.RED}Required file not found: {file_path}{color.ENDC}')
            
            # Load data with error handling
            data = np.load(file_path)
            
            # Validate data is not empty
            if data.size == 0:
                raise ValueError(f'{color.RED}Empty data file: {file_path}{color.ENDC}')
            
            # Validate data doesn't contain NaN or Inf
            if np.isnan(data).any():
                print(f'{color.RED}Warning: {file} contains NaN values. Replacing with 0.{color.ENDC}')
                data = np.nan_to_num(data, nan=0.0)
            
            if np.isinf(data).any():
                print(f'{color.RED}Warning: {file} contains Inf values. Clipping to finite range.{color.ENDC}')
                data = np.nan_to_num(data, posinf=1e10, neginf=-1e10)
            
            # Shape handling with validation
            if len(data.shape) == 2 and file != 'labels':
                print(f"📊 Reshaping {file} data from {data.shape} to add feature dimension")
                data = data[:, np.newaxis, :]  # Add feature dimension: [samples, 1, sequence]
            
            # Validate dimensions
            if file != 'labels' and len(data.shape) not in [2, 3]:
                raise ValueError(f'{color.RED}Invalid data shape for {file}: {data.shape}. Expected 2D or 3D.{color.ENDC}')
            
            print(f'{color.GREEN}Loaded {file}: shape={data.shape}, dtype={data.dtype}{color.ENDC}')
            loader.append(data)
            
        except FileNotFoundError as e:
            print(e)
            raise
        except Exception as e:
            print(f'{color.RED}Error loading {file}: {str(e)}{color.ENDC}')
            raise
    
    # Apply data reduction if requested
    if args.less: 
        loader[0] = cut_array(0.2, loader[0])
        print(f'{color.GREEN}Training data reduced to 20%: {loader[0].shape}{color.ENDC}')
    
    # Validate train/test/label alignment
    if loader[0].shape[0] == 0 or loader[1].shape[0] == 0:
        raise ValueError(f'{color.RED}Empty dataset after loading{color.ENDC}')
    
    # Create data loaders with error handling
    try:
        train_loader = DataLoader(loader[0], batch_size=64, shuffle=True, drop_last=False)
        test_loader = DataLoader(loader[1], batch_size=64, shuffle=False, drop_last=False)
    except Exception as e:
        print(f'{color.RED}Error creating data loaders: {str(e)}{color.ENDC}')
        raise
    
    labels = loader[2]
    
    print(f'{color.BOLD}Dataset loaded successfully: {dataset}{color.ENDC}')
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list, is_best=False):
    """
    Enhanced model saving with best model tracking.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        accuracy_list: Training history
        is_best: Whether this is the best model so far
    """
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    
    # Save checkpoint
    file_path = f'{folder}/model.ckpt'
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list
        }
        torch.save(checkpoint, file_path)
        print(f'{color.GREEN}Model checkpoint saved: {file_path}{color.ENDC}')
        
        # Save best model separately if this is the best
        if is_best:
            best_path = f'{folder}/best_model.ckpt'
            torch.save(checkpoint, best_path)
            print(f'{color.BOLD}Best model saved: {best_path}{color.ENDC}')
            
    except Exception as e:
        print(f'{color.RED}Error saving model: {str(e)}{color.ENDC}')
        raise


def load_model(modelname, dims):
    """
    Enhanced model loading with flexible learning rate scheduler options.
    
    Args:
        modelname: Name of the model class to load
        dims: Number of input dimensions/features
        
    Returns:
        model, optimizer, scheduler, epoch, accuracy_list
    """
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    
    # Enhanced learning rate scheduler options
    # Can be configured via args or model attributes
    scheduler_type = getattr(args, 'scheduler', 'step')  # Default to 'step'
    
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                 factor=0.5, patience=5, verbose=True)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        # Default to StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(fname, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Handle scheduler loading with fallback
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"{color.RED}Warning: Could not load scheduler state: {e}. Using fresh scheduler.{color.ENDC}")
            
            epoch = checkpoint['epoch']
            accuracy_list = checkpoint.get('accuracy_list', [])
            print(f"{color.GREEN}Loaded checkpoint from epoch {epoch}{color.ENDC}")
            
        except Exception as e:
            print(f"{color.RED}Error loading checkpoint: {e}. Creating new model.{color.ENDC}")
            epoch = -1
            accuracy_list = []
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    # device = torch.device('cpu')  # Force CPU usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tqdm.write(f"Using device: {device}")
    data = data.to(device, dtype=torch.float64)
    dataO = dataO.to(device, dtype=torch.float64)
    model = model.to(device)
    
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(device)
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                d = d.to(device)
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    if 'Attention' in model.name: 
        l = nn.MSELoss(reduction='none')
        model.to(device)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        res = []
        if training:
            for d in data:
                d = d.to(device)
                ae = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            model.to(device)
            for i, d in enumerate(data):
                d = d.to(device)
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item());
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(device)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.to(device)
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        model.to(device)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                d = d.to(device)
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            xs = []
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(device)
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1;
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                d = d.to(device)
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(device)
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], [] 
        if training:
            for d, _ in dataloader:  
                d = d.to(device)
                local_bs = d.shape[0] 
                window = d.permute(1, 0, 2)  
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)  
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'DTAAD' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        model.to(device)
        # data assumed shape: [num_windows, window_size, num_features]
        data_x = data.to(torch.float64)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]
                # Add debug print
                # print(f"🔍 DTAAD backprop input d shape: {d.shape}")
                # window = d.permute(0, 2, 1)  # [batch, features, window_size]
                if d.shape[1] == 1 and d.shape[2] == 10:  # [batch, 1, 10] - already correct format
                    window = d  # No permutation needed
                else:
                    window = d.permute(0, 2, 1)  # [batch, features, window_size]
            
                # print(f"🔍 DTAAD window shape: {window.shape}")
                num_features = window.shape[1]
                elem = window[:, :, -1].view(1, local_bs, num_features)  # [1, batch, features]
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2), elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:  # Testing phase
            model.to(device)
            for d, _ in dataloader:
                d = d.to(device)
                
                # Same dimension fix as training
                if len(d.shape) == 3 and d.shape[2] == 10:  # [batch, features, window_size]
                    window = d  # Already in correct format
                    num_features = window.shape[1]
                else:
                    window = d.permute(0, 2, 1)
                    num_features = window.shape[1]
                
                local_bs = d.shape[0]
                elem = window[:, :, -1].view(1, local_bs, num_features)  # [1, batch, features]
                z = model(window)
                z = z[1].permute(1, 0, 2)
            loss = l(z, elem)[0]
            loss=loss.to('cpu')
            return loss.detach().numpy(), z.detach().cpu().numpy()[0]
    else:
        model.to(device)
        data = data.to(device)
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(device)
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    # train_loader, test_loader, labels = load_dataset(args.dataset)
    # # model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])
    # train_loader, test_loader, labels = load_dataset(args.dataset)
    # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    # model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, trainD.shape[1])  # Use data features, not label features

    # ## Prepare data
    # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    # trainO, testO = trainD, testD
    # # OLD:
    # # if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
    # #                   'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
    # #     trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    # # NEW: Use updated convert_to_windows, outputs [num_windows, window_size, num_features]
    # if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
    #                   'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
    #     trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device detected: {device}")
    train_loader, test_loader, labels = load_dataset(args.dataset)
    
    # Get sample data to determine feature count
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    num_features = trainD.shape[1]
    print(f"DEBUG: Number of features detected: {num_features}")
    
    # Use actual data features, not label features
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, num_features)

    ## Prepare data (no need to call next(iter()) again - already have trainD, testD)
    trainO, testO = trainD, testD
    
    # Convert to windows for models that need it
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                        'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 20
        e = epoch + 1
        
        # Initialize early stopping with configurable patience
        patience = getattr(args, 'early_stopping_patience', 7)
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, verbose=True, mode='min')
        best_loss = float('inf')
        
        print(f'{color.GREEN}Early stopping patience: {patience} epochs{color.ENDC}')
        print(f'{color.GREEN}Learning rate scheduler: {getattr(args, "scheduler", "step")}{color.ENDC}')
        
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
            
            # Check for best model
            is_best = lossT < best_loss
            if is_best:
                best_loss = lossT
            
            # Save model (will save best if is_best=True)
            if e % 5 == 0 or is_best:  # Save every 5 epochs or if best
                save_model(model, optimizer, scheduler, e, accuracy_list, is_best=is_best)
            
            # Check early stopping
            if early_stopping(lossT, e):
                print(f'{color.BOLD}Early stopping triggered at epoch {e}{color.ENDC}')
                break
        
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        
        # Final save
        save_model(model, optimizer, scheduler, e, accuracy_list, is_best=False)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### Plot curves
    # if not args.test:
    #     if 'TranAD' or 'DTAAD' in model.name: testO = torch.roll(testO, 1, 0)
    #     plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
    ### Plot curves
    if not args.test:
        # print("📊 Preparing data for plotting...")
        
        # Convert tensors to numpy and handle shape mismatch
        if isinstance(testO, torch.Tensor):
            testO_np = testO.detach().cpu().numpy()
        else:
            testO_np = testO
        
        # Take first sample and first channel for plotting: (48, 1, 17479) -> (17479,)
        if len(testO_np.shape) == 3:
            testO_plot = testO_np[0, 0, :]  # First sample, first channel
        else:
            testO_plot = testO_np[0] if len(testO_np.shape) > 1 else testO_np
        
        # For y_pred, take first portion that matches original length
        y_pred_plot = y_pred[:len(testO_plot)] if len(y_pred) > len(testO_plot) else y_pred
        ascore_plot = loss[:len(testO_plot)] if len(loss) > len(testO_plot) else loss
        
        # Create dummy labels for plotting (same length as testO_plot) - FIXED VERSION
        labels_plot = np.zeros((len(testO_plot), 1))
        if len(labels.shape) > 1 and labels.shape[0] > 0:
            # Use simpler approach - tile the labels to match length
            repeat_factor = len(testO_plot) // len(labels) + 1
            repeated_labels = np.tile(labels[:, 0], repeat_factor)
            labels_plot[:, 0] = repeated_labels[:len(testO_plot)]  # Trim to exact length
        
        # Ensure all have same length
        min_len = min(len(testO_plot), len(y_pred_plot), len(ascore_plot))
        testO_plot = testO_plot[:min_len].reshape(-1, 1)
        y_pred_plot = y_pred_plot[:min_len].reshape(-1, 1)
        ascore_plot = ascore_plot[:min_len].reshape(-1, 1)
        labels_plot = labels_plot[:min_len].reshape(-1, 1)
        
        # print(f"🎯 Aligned shapes: testO: {testO_plot.shape}, y_pred: {y_pred_plot.shape}, ascore: {ascore_plot.shape}, labels: {labels_plot.shape}")
        
        if 'TranAD' in model.name or 'DTAAD' in model.name: 
            testO_plot = np.roll(testO_plot, 1, 0)
        
        plotter(f'{args.model}_{args.dataset}', testO_plot, y_pred_plot, ascore_plot, labels_plot)

    ### Plot attention
    if not args.test:
        if 'DTAAD' in model.name:
            plot_attention(model, 1, f'{args.model}_{args.dataset}')

    ### Scores
    df = pd.DataFrame()
    preds = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    # print(f"🔍 Debug shapes before scoring:")
    # print(f"  lossT shape: {lossT.shape}")
    # print(f"  loss shape: {loss.shape}")
    # print(f"  labels shape: {labels.shape}")

    # # FIX: Create windowed labels to match windowed predictions
    # print("🔧 Creating windowed labels to match predictions...")

    # Convert original labels to windowed format
    if len(labels.shape) == 2 and labels.shape[0] < loss.shape[0]:
        # Expand labels to match windowed data
        # Each original sample becomes multiple windows
        original_samples = labels.shape[0]
        windows_per_sample = loss.shape[0] // original_samples
        
        # Repeat each label for all its windows
        windowed_labels = np.repeat(labels, windows_per_sample, axis=0)
        
        # Trim to exact length if needed
        windowed_labels = windowed_labels[:loss.shape[0]]
        
        # print(f"🎯 Windowed labels shape: {windowed_labels.shape}")
    else:
        windowed_labels = labels

    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], windowed_labels[:, i]
        # print(f"🔍 Dimension {i}:")
        # print(f"  lt (train loss) length: {len(lt)}")
        # print(f"  l (test loss) length: {len(l)}")
        # print(f"  ls (labels) length: {len(ls)}")
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    # labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    # Create final labels by averaging windowed labels back to original samples
    if len(windowed_labels.shape) > 1:
        labelsFinal = np.mean(windowed_labels, axis=1)
        labelsFinal = (labelsFinal >= 0.5).astype(int)  # Threshold back to binary
    else:
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    # print(f"🎯 Final scoring shapes:")
    # print(f"  lossTfinal: {lossTfinal.shape}")
    # print(f"  lossFinal: {lossFinal.shape}")  
    # print(f"  labelsFinal: {labelsFinal.shape}")
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, windowed_labels))
    result.update(ndcg(loss, windowed_labels))
    
    # Calculate additional F1, precision, recall metrics
    print(f'\n{color.HEADER}Enhanced Metrics Calculation{color.ENDC}')
    
    # Convert anomaly scores to binary predictions using optimal threshold
    threshold = result.get('threshold', np.percentile(lossFinal, 95))
    y_pred_binary = (lossFinal > threshold).astype(int)
    
    # Calculate F1, precision, recall
    enhanced_metrics = calculate_f1_precision_recall(labelsFinal, y_pred_binary)
    result.update({
        'f1_sklearn': enhanced_metrics['f1'],
        'precision_sklearn': enhanced_metrics['precision'],
        'recall_sklearn': enhanced_metrics['recall']
    })
    
    # Calculate precision-recall curve
    pr_curve = calculate_precision_recall_curve(labelsFinal, lossFinal)
    result.update({'pr_auc': pr_curve['pr_auc']})
    
    print(f'{color.BOLD}Per-dimension Results:{color.ENDC}')
    print(df)
    print(f'\n{color.BOLD}Overall Results:{color.ENDC}')
    pprint(result)
    print(f'\n{color.GREEN}Enhanced Metrics:{color.ENDC}')
    print(f'  F1 Score (sklearn): {enhanced_metrics["f1"]:.4f}')
    print(f'  Precision (sklearn): {enhanced_metrics["precision"]:.4f}')
    print(f'  Recall (sklearn): {enhanced_metrics["recall"]:.4f}')
    print(f'  PR-AUC: {pr_curve["pr_auc"]:.4f}')
    # pprint(getresults2(df, result))