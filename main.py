import pandas as pd
import numpy as np
import pickle
import math
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence


nt_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}

class Config:
    vocab_size = len(nt_to_idx)
    max_len = 512
    emb_dim = 128
    num_layers = 6
    nhead = 4
    ff_dim = 512
    dropout = 0.4
    batch_size = 2
    epochs = 30
    lr = 5e-3
    pad_idx = 4
    
    use_scheduler = True
    seed = 42

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
config.set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

class RNAStruct(Dataset):
    def __init__(self, seq_csv, labels_csv, mean=None, std=None, normalize=True):
        seq_df = pd.read_csv(seq_csv)
        seq_map = {row['target_id']: row['sequence'].strip() for _, row in seq_df.iterrows()}

        labels_df = pd.read_csv(labels_csv)
        self.data = {}
        for _, row in labels_df.iterrows():
            target_id = row['ID'].rsplit('_', 1)[0]
            idx = int(row['resid']) - 1
            if target_id not in self.data:
                self.data[target_id] = []
            self.data[target_id].append((idx, row['resname'], row['x_1'], row['y_1'], row['z_1']))

        self.samples = []
        for target_id in self.data:
            dat = sorted(self.data[target_id], key=lambda x: x[0])
            seq = seq_map[target_id]
            coords = np.array([[x[2], x[3], x[4]] for x in dat], dtype=np.float32)
            if len(seq) == len(coords) and np.isfinite(coords).all():
                seq_idx = np.array([nt_to_idx.get(nt, 4) for nt in seq], dtype=np.int64)
                self.samples.append((target_id, seq_idx, coords))
            else:
                print(f"Bad entry removed: {target_id} (len/coords/finite)")

        all_coords = np.concatenate([coords for _, _, coords in self.samples], axis=0)
        self.mean = mean if mean is not None else all_coords.mean(axis=0)
        self.std = std if std is not None else all_coords.std(axis=0)
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target_id, seq_idx, coords = self.samples[idx]
        length = len(seq_idx)
        if self.normalize:
            if isinstance(self.mean, torch.Tensor):
                mean = self.mean.numpy()
            else:
                mean = self.mean
            if isinstance(self.std, torch.Tensor):
                std = self.std.numpy()
            else:
                std = self.std
            norm_coords = (coords - mean) / std
            return (
        torch.LongTensor(seq_idx),
        torch.tensor(norm_coords, dtype=torch.float32),
        length,
        target_id
    ) 
    def get_mean_std(self,):
        return self.mean, self.std
    
class PositionalEncoding(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, x):
            seq_len = x.size(1)
            device = x.device
            position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model)
        )
            pe = torch.zeros(seq_len, self.d_model, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)

#collate function
def rna_collate(batch):
    seqs, coords, lengths, target_ids = zip(*batch)
    lengths = torch.tensor(lengths)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=4)
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0)
    return seqs_padded, coords_padded, lengths, target_ids


# Model
class RNA3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_idx)
        self.pos_enc = PositionalEncoding(config.emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim, nhead=config.nhead, dim_feedforward=config.ff_dim,
            dropout=config.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.ln = nn.LayerNorm(config.emb_dim)
        self.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.emb_dim, 3)) 
    def forward(self, seq, lengths, noise_std=0.0):
        x = self.embed(seq)
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        x = self.pos_enc(x)
        mask = (seq == 4) 
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.ln(x)
        coords = self.fc(x)
        return coords

def compute_tm_score(pred_coords, true_coords, Lref=None):
    assert pred_coords.shape == true_coords.shape
    L = pred_coords.shape[0]
    Lref = Lref if Lref is not None else L
    dists = np.linalg.norm(pred_coords - true_coords, axis=1)
    
    # d0 depends on Lref
    if Lref >= 30:
        d0 = 1.24 * (Lref - 15) ** (1/3) - 1.8
    elif Lref >= 24:
        d0 = 0.7
    elif Lref >= 20:
        d0 = 0.6
    elif Lref >= 16:
        d0 = 0.5
    elif Lref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3

    score = (1 / Lref) * np.sum(1 / (1 + (dists / d0) ** 2))
    return score

def kabsch_align(P, Q):
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = np.dot(V, Wt)
    return np.dot(P, U)



def validate(model, dataloader, device):
    model.eval()
    total_pts, total_rmsd, total_tm = 0, 0, 0
    with torch.no_grad():
        for seqs, coords, lengths, _ in dataloader:
            seqs, coords, lengths = seqs.to(device), coords.to(device), lengths.to(device)
            pred_coords = model(seqs, lengths)
            for i in range(seqs.size(0)):
                L = lengths[i].item()
                pred = pred_coords[i, :L].cpu().numpy()
                true = coords[i, :L].cpu().numpy()

                pred_centered = pred - pred.mean(axis=0)
                true_centered = true - true.mean(axis=0)
                aligned_pred = kabsch_align(pred_centered, true_centered)
                dists = np.linalg.norm(aligned_pred - true_centered, axis=1)
                rmsd = np.sqrt((dists ** 2).mean())
                tm = compute_tm_score(aligned_pred, true_centered, Lref=L)

                total_rmsd += rmsd * L
                total_tm += tm * L
                total_pts += L

    mean_rmsd = total_rmsd / total_pts
    mean_tm = total_tm / total_pts
    model.train()
    return mean_rmsd, mean_tm

# training loop
def train_model(model, dataloader, device, val_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) #adamW
    num_training_steps = config.epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    #CosineAnnealingLR(optimizer, T_max=100) #ReduceLROnPlateau(optimizer, 'min', patience=2)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)if config.use_scheduler else None
    #loss_fn = nn.MSELoss(reduction='none')
    best_val = float('inf')

    epoch_losses = []
    val_scores = []

    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total_points = 0
        
        for seqs, coords, lengths, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            seqs, coords, lengths = seqs.to(device), coords.to(device), lengths.to(device)
            optimizer.zero_grad()
            pred_coords = model(seqs, lengths)
            mse_per_nt = ((pred_coords - coords) ** 2).sum(-1)    
            mask = (seqs != config.pad_idx)
            loss = (mse_per_nt * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * mask.sum().item()
            total_points += mask.sum().item()
            global_step += 1
        epoch_loss = running_loss / total_points
        epoch_losses.append(epoch_loss)

        # Print loss after each epoch
        print(f"\nEpoch {epoch+1} Loss: {epoch_loss:.5f}")

        if val_loader is not None:
            val_rmsd, _ = validate(model, val_loader, device)
            val_scores.append(val_rmsd)  
    
            print(f"Validation RMSD: {val_rmsd:.5f}")

            if scheduler is not None:
                scheduler.step()
            if val_rmsd < best_val:
                best_val = val_rmsd
                torch.save(model.state_dict(), 'setnet.pt')
                val_rmsd, val_tm = validate(model, val_loader, device)
                print(f"New BEST model saved at Epoch {epoch+1} (RMSD: {val_rmsd:.4f})")
                print(f"\nFinal TM-score on validation set: {val_tm:.4f}")
                print(f"Final RMSD on validation set: {val_rmsd:.4f}")
                 
    # plot losses and validation scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    if val_loader is not None:
        plt.subplot(1, 2, 2)
        plt.plot(val_scores, label='Validation RMSD', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('RMSD')
        plt.title('Validation RMSD Over Epochs')
        plt.legend()

    plt.tight_layout()
    plt.show()

