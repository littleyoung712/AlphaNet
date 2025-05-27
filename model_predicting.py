from model_training import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models import AlphaNet
import pandas as pd
from typing import Optional, Union, List
import os


class PredictionDataset(Dataset):
    """用于预测的数据集类"""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

def load_model(model_path: str, device: str = 'cuda') -> AlphaNet:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    # 模型参数
    parameters = {
        'window': 10,
        'stride': 10,
        'n_features': 8
    }
    
    model = AlphaNet(d=parameters['window'], 
                    stride=parameters['stride'], 
                    n=parameters['n_features'])
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model: AlphaNet, 
           data_loader: DataLoader,
           device: str = 'cuda') -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch,y_real in tqdm(data_loader, desc='模型预测中'):
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)




if __name__ == "__main__":
    model_path = "Models/AlphaNet_batch_size1000.pt"
    alphanet = load_model(model_path)
    prediction_train = predict(alphanet, train_loader, device='cuda')
    prediction_valid = predict(alphanet, valid_loader, device='cuda')
    prediction_test = predict(alphanet, test_loader, device='cuda')
    prediction_train_sr = pd.DataFrame(prediction_train,index=sorted_indices.loc[train_index])
    prediction_valid_sr = pd.DataFrame(prediction_valid,index=sorted_indices.loc[val_index])
    prediction_test_sr = pd.DataFrame(prediction_test,index=sorted_indices.loc[test_index])
    prediction_sr = pd.concat([prediction_train_sr,prediction_valid_sr,prediction_test_sr], axis=0)
    prediction_sr['st_code'] = prediction_sr.index.map(lambda x: x[0])
    prediction_sr['trade_date'] = prediction_sr.index.map(lambda x: x[1])
    prediction_sr = prediction_sr.reset_index(drop=True).rename(columns={0:'prediction'})
    prediction_sr.to_csv('result/model_prediction1.csv')


