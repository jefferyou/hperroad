"""
高度优化的TTE评估脚本
- 预加载embedding到GPU避免重复计算
- 多GPU并行训练 (DataParallel)
- 优化数据加载pipeline
"""
import sys
import os

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
veccity_path = os.path.join(project_root, 'VecCity-main')
os.chdir(veccity_path)
sys.path.insert(0, veccity_path)

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from logging import getLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--task_epoch', type=int, default=10)
    parser.add_argument('--gpu_ids', type=str, default='3,4,5,6,7',
                        help='Comma-separated GPU IDs to use (e.g., "3,4,5,6,7")')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (will be split across GPUs)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader workers')
    return parser.parse_args()

class PreloadedEmbeddingModel(nn.Module):
    """预加载embedding的模型，避免重复计算"""
    def __init__(self, embeddings_tensor, device):
        super().__init__()
        # 将embedding注册为buffer（不参与训练但会跟随模型移动）
        self.register_buffer('embeddings', embeddings_tensor)
        self.device = device

    def encode(self, x):
        """直接从预加载的embedding中索引"""
        return self.embeddings[x.long()]

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                           dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, path, valid_len):
        original_shape = path.shape
        full_embed = self.embedding.encode(path)
        full_embed = full_embed.view(*original_shape, self.input_dim)
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len.cpu(),
                                     batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim, device=self.device)
        _, out = self.lstm(pack_x, (h0, c0))
        return out[0][0]

class MLPReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, embedding, device):
        super(MLPReg, self).__init__()
        self.embedding = embedding
        self.lstm = TrajEncoder(input_dim, input_dim, 1, embedding, device)

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()

    def forward(self, path, valid_len):
        x = self.lstm(path, valid_len)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x.squeeze(-1)

class TTEDataset(Dataset):
    def __init__(self, X, lens, y):
        self.X = X
        self.lens = lens
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.lens[idx], self.y[idx]

def run_optimized_tte(args):
    print("=" * 80)
    print("OPTIMIZED TTE EVALUATION with Multi-GPU")
    print(f"Experiment ID: {args.exp_id}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Workers: {args.num_workers}")
    print("=" * 80)

    # 解析GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')

    # 1. 加载预计算的embedding到GPU
    embedding_path = f'veccity/cache/{args.exp_id}/evaluate_cache/road_embedding_HRNR_Hyperbolic_xa_128.npy'
    print(f"\nLoading embeddings from: {embedding_path}")
    emb = np.load(embedding_path)
    print(f"✓ Loaded embeddings shape: {emb.shape}")

    # 转换为torch tensor并移到GPU
    emb_tensor = torch.from_numpy(emb).float().to(primary_device)
    print(f"✓ Embeddings loaded to {primary_device}, size: {emb_tensor.shape}")

    # 创建预加载embedding模型
    embedding_model = PreloadedEmbeddingModel(emb_tensor, primary_device)

    # 2. 加载TTE数据
    print("\nLoading TTE data...")
    import pickle
    with open(f'./veccity/cache/dataset_cache/xa/time_estimation_data/time_estimation_data.pkl', 'rb') as f:
        eta_data = pickle.load(f)

    num_samples = len(eta_data)
    max_len = max([len(row['path']) for row in eta_data])

    x_arr = np.zeros((num_samples, max_len))
    lens_arr = np.zeros(num_samples)
    y_arr = np.zeros(num_samples)

    for i, row in enumerate(eta_data):
        path = row['path']
        lens = len(path)
        if lens < max_len:
            path = np.append(path, [0] * (max_len - lens))
        x_arr[i,:] = path
        lens_arr[i] = lens
        y_arr[i] = row['time']

    x_arr = torch.Tensor(x_arr).long()
    lens_arr = torch.Tensor(lens_arr).long()
    y_arr = torch.Tensor(y_arr)

    # 划分数据集
    train_size = int(num_samples * 0.6)
    eval_size = int(num_samples * 0.2)

    train_dataset = TTEDataset(x_arr[:train_size], lens_arr[:train_size], y_arr[:train_size])
    eval_dataset = TTEDataset(x_arr[train_size:train_size+eval_size],
                             lens_arr[train_size:train_size+eval_size],
                             y_arr[train_size:train_size+eval_size])
    test_dataset = TTEDataset(x_arr[train_size+eval_size:],
                             lens_arr[train_size+eval_size:],
                             y_arr[train_size+eval_size:])

    print(f"✓ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")

    # 3. 创建DataLoader with优化
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True if args.num_workers > 0 else False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True if args.num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True if args.num_workers > 0 else False)

    # 4. 创建模型并包装为DataParallel
    input_dim = emb.shape[1]
    hidden_dim = 128
    model = MLPReg(input_dim, hidden_dim, 2, embedding_model, primary_device)

    # 使用DataParallel包装模型以使用多GPU
    if len(gpu_ids) > 1:
        print(f"\n✓ Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(primary_device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print("\n" + "=" * 80)
    print(f"Starting Training for {args.task_epoch} epochs")
    print("=" * 80)

    best = {"best_epoch": 0, "mae": 1e9, "rmse": 1e9}
    patience = 10

    for epoch in range(args.task_epoch):
        # 训练
        model.train()
        train_loss = 0
        for batch_x, batch_lens, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.task_epoch}"):
            batch_x = batch_x.to(primary_device, non_blocking=True)
            batch_lens = batch_lens.to(primary_device, non_blocking=True)
            batch_y = batch_y.to(primary_device, non_blocking=True)

            opt.zero_grad()
            preds = model(batch_x, batch_lens)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 评估
        model.eval()
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for batch_x, batch_lens, batch_y in eval_loader:
                batch_x = batch_x.to(primary_device, non_blocking=True)
                batch_lens = batch_lens.to(primary_device, non_blocking=True)
                y_preds.append(model(batch_x, batch_lens).cpu())
                y_trues.append(batch_y)

        y_preds = torch.cat(y_preds, dim=0).numpy()
        y_trues = torch.cat(y_trues, dim=0).numpy()

        mae = mean_absolute_error(y_trues, y_preds)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        if mae < best["mae"]:
            best = {"best_epoch": epoch+1, "mae": mae, "rmse": rmse}
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\n" + "=" * 80)
    print(f"Best Epoch: {best['best_epoch']}, MAE: {best['mae']:.4f}, RMSE: {best['rmse']:.4f}")
    print("=" * 80)

    return best

if __name__ == '__main__':
    args = parse_args()
    run_optimized_tte(args)
