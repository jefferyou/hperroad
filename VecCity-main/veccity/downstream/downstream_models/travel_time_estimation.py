import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm
from veccity.downstream.downstream_models.abstract_model import AbstractModel
from torch.utils.data import Dataset, DataLoader
import random

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        # 获取实际的embedding维度（支持EmbeddingWrapper和numpy数组）
        if hasattr(embedding, 'shape'):
            actual_embed_dim = embedding.shape[1] if len(embedding.shape) > 1 else embedding.shape[0]
        else:
            actual_embed_dim = input_dim

        self.input_dim = actual_embed_dim  # 使用实际维度而不是配置中的维度
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(actual_embed_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, path, valid_len):

        original_shape = path.shape  # [batch_size, traj_len]
        full_embed = self.embedding.encode(path)
        full_embed = full_embed.view(*original_shape, self.input_dim)  # [batch_size, traj_len, embed_size]
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        _, out = self.lstm(pack_x, (h0, c0))

        return out[0][0]


class MLPReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, embedding, is_static,device, max_len):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.embedding = embedding
        self.lstm = TrajEncoder(input_dim, input_dim, 1, embedding, device)

        # MLPReg的输入是LSTM的输出维度（hidden_dim），而不是embedding维度
        # TrajEncoder的输出维度是它的hidden_dim参数
        lstm_output_dim = self.lstm.hidden_dim  # LSTM输出维度（128），不是embedding维度（225）

        self.layers = []
        self.layers.append(nn.Linear(lstm_output_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, path, valid_len,**kwargs):
        
        x=self.lstm(path,valid_len,**kwargs)

        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


class TimeEstimationDataset(Dataset):
    def __init__(self, data_x,data_lens, data_y):
        self.x = data_x
        self.lens = data_lens
        self.y = data_y
        assert len(self.x) == len(self.y)

    def __getitem__(self, item):
        return self.x[item],self.lens[item], self.y[item]

    def __len__(self):
        return len(self.x)

class TravelTimeEstimationModel(AbstractModel):
    def __init__(self, config):
        self.config=config
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './veccity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)


    def run(self, embedding_model, label, embed_model=None,**kwargs):
        self._logger.info("--- Time Estimation ---")
        min_len, max_len = self.config.get("tte_min_len", 10), self.config.get("tte_max_len", 128)
        dfs = label['time']
        num_samples = len(dfs)
        padding_id = 0
        x_arr = np.zeros([num_samples, max_len],dtype=np.int64)
        lens_arr = np.zeros([num_samples], dtype=np.int64)
        y_arr = np.zeros([num_samples], dtype=np.float32)
        for i in range(num_samples):
            row = dfs.iloc[i]
            path = [int(x) for x in row['path']]
            lens = len(path)

            if len(row['path']) < max_len:
                temp = np.array([padding_id] * (max_len - len(row['path'])))
                path = np.append(path, temp)

            path = np.array(path)
            x_arr[i,:] = path
            lens_arr[i] = lens
            y_arr[i] = row['time']
        x_arr = torch.Tensor(x_arr).long()
        lens_arr = torch.Tensor(lens_arr).long()
        y_arr = torch.Tensor(y_arr)

        # Normalize travel time for better training stability
        y_mean = y_arr.mean()
        y_std = y_arr.std()
        y_arr_normalized = (y_arr - y_mean) / (y_std + 1e-8)

        device=self.config.get('device','cpu')
        input_dim=self.config.get('embed_size',128)
        hidden_dim=self.config.get('d_model', 128)
        max_epoch = self.config.get('task_epoch',100)
        is_static = self.config.get('is_static',True)
        train_size = int(num_samples * 0.6)
        eval_size=int(num_samples * 0.2)
        test_size = num_samples - train_size - eval_size
        train_data_X ,train_lens,train_data_y = x_arr[:train_size],lens_arr[:train_size],y_arr_normalized[:train_size]
        eval_data_X ,eval_lens, eval_data_y = x_arr[train_size:train_size+eval_size],lens_arr[train_size:train_size+eval_size],y_arr_normalized[train_size:train_size+eval_size]
        test_data_X ,test_lens, test_data_y = x_arr[train_size+eval_size:],lens_arr[train_size+eval_size:],y_arr_normalized[train_size+eval_size:]   

        train_dataset = TimeEstimationDataset(train_data_X,train_lens,train_data_y)
        eval_dataset = TimeEstimationDataset(eval_data_X,eval_lens,eval_data_y)
        test_dataset = TimeEstimationDataset(test_data_X,test_lens,test_data_y)

        train_dataloader= DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=4)
        eval_dataloader= DataLoader(eval_dataset,batch_size=128,shuffle=False,num_workers=4)
        test_dataloader= DataLoader(test_dataset,batch_size=128,shuffle=False,num_workers=4)


        model = MLPReg(input_dim, hidden_dim, 2, nn.ReLU(), embedding_model,is_static,device,max_len).to(device)

        # Configurable learning rate with default 1e-3 (higher than previous 1e-4)
        learning_rate = self.config.get('tte_learning_rate', 1e-3)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )

        loss_fn = nn.MSELoss()

        # Configurable patience for early stopping (default 30, increased from 10)
        patience = self.config.get('tte_patience', 30)
        initial_patience = patience  # Store initial patience for reset

        # Gradient clipping threshold
        max_grad_norm = self.config.get('max_grad_norm', 1.0)

        best = {"best epoch": 0, "mae": 1e9, "rmse": 1e9}

        for epoch in range(max_epoch):
            model.train()
            for batch_x,batch_lens,batch_y in tqdm(train_dataloader):
                opt.zero_grad()

                batch_x = batch_x.to(device)
                batch_lens = batch_lens.to(device)
                batch_y = batch_y.to(device)
                preds=model(batch_x,batch_lens,**kwargs)
                loss = loss_fn(preds, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                opt.step()

            model.eval()
            y_preds = []
            y_trues = []
            for batch_x, batch_lens,batch_y in eval_dataloader:
                batch_x = batch_x.to(device)
                batch_lens = batch_lens.to(device)
                batch_y = batch_y.to(device)
                y_preds.append(model(batch_x,batch_lens,**kwargs).detach().cpu())
                y_trues.append(batch_y.detach().cpu())
            
            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)

            # Denormalize predictions for evaluation
            y_preds_denorm = y_preds * y_std + y_mean
            y_trues_denorm = y_trues * y_std + y_mean

            mae = mean_absolute_error(y_trues_denorm, y_preds_denorm)
            rmse = mean_squared_error(y_trues_denorm, y_preds_denorm) ** 0.5
            self._logger.info(f'Epoch: {epoch}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')

            # Update learning rate based on validation MAE
            scheduler.step(mae)

            # self._writer.add_scalar('ETA Valid MAE', mae, epoch)
            # self._writer.add_scalar('ETA Valid RMSE', rmse, epoch)
            if mae < best["mae"]:
                best = {"best epoch": epoch, "mae": mae, "rmse": rmse}
                patience = initial_patience  # Reset to initial patience value
                # best_model=copy.deepcopy(model)
            else:
                patience -= 1
                if not patience:
                    self._logger.info("Early stopping triggered. Best epoch: {}, MAE:{}, RMSE:{}".format(best['best epoch'], best['mae'], best["rmse"]))
                    break

        model.eval()
        y_preds = []
        y_trues = []
        for batch_x, batch_lens,batch_y in test_dataloader:
                batch_x = batch_x.to(device)
                batch_lens = batch_lens.to(device)
                batch_y = batch_y.to(device)
                y_preds.append(model(batch_x,batch_lens,**kwargs).detach().cpu())
                y_trues.append(batch_y.detach().cpu())
        
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)

        # Denormalize predictions for final evaluation
        y_preds_denorm = y_preds * y_std + y_mean
        y_trues_denorm = y_trues * y_std + y_mean

        mae = mean_absolute_error(y_trues_denorm, y_preds_denorm)
        rmse = mean_squared_error(y_trues_denorm, y_preds_denorm) ** 0.5
        best['mae']=mae
        best['rmse']=rmse
        self._logger.info("Test result:epoch {}, MAE:{}, RMSE:{}".format(best['best epoch'], best['mae'], best["rmse"]))
        # self._writer.close()
        return best

    def clear(self):
        pass

    def save_result(self, save_path, filename=None):
        pass