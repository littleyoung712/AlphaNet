import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr


# --------------------特征提取层--------------------

class ts_corr(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
    """

    def __init__(self, d=10, stride=10):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_corr, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        # n-特征数量，T-时间窗口
        batch_size, n, T = X.shape

        # 初始化输出特征图
        w = int((T - self.d) / self.stride + 1)
        h = int(n * (n - 1) / 2)
        Z = torch.zeros(batch_size, h, w, device=X.device)

        # 使用向量化操作替代循环
        for i in range(w):
            start = i * self.stride
            end = start + self.d
            
            # 获取当前时间窗口的所有特征
            window = X[:, :, start:end]  # [batch_size, n, d]
            start_idx = 0
            # 计算所有特征对之间的相关系数
            for j in range(n - 1):
                # 获取当前特征
                x = window[:, j:j+1, :]  # [batch_size, 1, d]
                # 获取剩余特征
                y = window[:, j+1:, :]   # [batch_size, n-j-1, d]
                
                # 计算相关系数
                x_mean = x.mean(dim=2, keepdim=True)
                y_mean = y.mean(dim=2, keepdim=True)
                
                x_std = torch.std(x, dim=2, keepdim=True)
                y_std = torch.std(y, dim=2, keepdim=True)
                
                # 计算协方差
                cov = ((x - x_mean) * (y - y_mean)).mean(dim=2)
                
                # 计算相关系数
                corr = cov / (x_std * y_std).squeeze(-1)
                
                # 更新特征图

                end_idx = start_idx + (n - j - 1)
                Z[:, start_idx:end_idx, i] = corr
                start_idx = end_idx
        return Z


class ts_cov(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
    """

    def __init__(self, d=10, stride=10,):
        super(ts_cov, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        h = int(n * (n - 1) / 2)
        Z = torch.zeros(batch_size, h, w, device=X.device)

        for i in range(w):
            start = i * self.stride
            end = start + self.d
            
            # 获取当前时间窗口的所有特征
            window = X[:, :, start:end]  # [batch_size, n, d]
            start_idx = 0
            for j in range(n - 1):
                # 获取当前特征
                x = window[:, j:j+1, :]  # [batch_size, 1, d]
                # 获取剩余特征
                y = window[:, j+1:, :]   # [batch_size, n-j-1, d]
                
                # 计算均值
                x_mean = x.mean(dim=2, keepdim=True)
                y_mean = y.mean(dim=2, keepdim=True)
                
                # 计算协方差
                cov = ((x - x_mean) * (y - y_mean)).mean(dim=2)
                
                # 更新特征图
                end_idx = start_idx + (n - j - 1)
                Z[:, start_idx:end_idx, i] = cov
                start_idx = end_idx

        return Z


class ts_return(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的return
    """

    def __init__(self, d=10, stride=10,):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_return, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        # n-特征数量，T-时间窗口
        batch_size, n, T = X.shape

        # 初始化输出特征图
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w, device=X.device)

        # 使用向量化操作计算return
        for i in range(w):
            start = i * self.stride
            end = start + self.d
            window = X[:, :, start:end]  # [batch_size, n, d]
            
            # 计算窗口的return
            return_d = (window[:, :, -1] - window[:, :, 0]) / window[:, :, 0]
            
            # 更新特征图
            Z[:, :, i] = return_d

        return Z


class ts_stddev(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的标准差
    """

    def __init__(self, d=10, stride=10,):
        super(ts_stddev, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w, device=X.device)

        # 使用向量化操作计算标准差
        for i in range(w):
            start = i * self.stride
            end = start + self.d
            window = X[:, :, start:end]  # [batch_size, n, d]
            
            # 计算标准差
            std = torch.std(window, dim=2)
            
            # 更新特征图
            Z[:, :, i] = std

        return Z


class ts_zscore(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的z-score
    """

    def __init__(self, d=10, stride=10,):
        super(ts_zscore, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w, device=X.device)

        # 使用向量化操作计算z-score
        for i in range(w):
            start = i * self.stride
            end = start + self.d
            window = X[:, :, start:end]  # [batch_size, n, d]
            
            # 计算均值和标准差
            mean = torch.mean(window, dim=2)
            std = torch.std(window, dim=2)
            
            # 计算z-score
            z_score = mean / (std + 1e-8)  # 添加小量防止除零
            
            # 更新特征图
            Z[:, :, i] = z_score

        return Z


class ts_decaylinear(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的加权平均值
    """

    def __init__(self, d=10, stride=10,):
        super(ts_decaylinear, self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w, device=X.device)

        # 计算权重
        weights = torch.arange(1, self.d + 1, device=X.device)
        normalized_w = weights / torch.sum(weights)

        # 使用向量化操作计算加权平均值
        for i in range(w):
            start = i * self.stride
            end = start + self.d
            window = X[:, :, start:end]  # [batch_size, n, d]
            
            # 计算加权平均值
            weighted_avg = torch.matmul(window, normalized_w)
            
            # 更新特征图
            Z[:, :, i] = weighted_avg

        return Z


# --------------------AlphaNet-v1--------------------

class AlphaNet(nn.Module):
    '''
    第一版AlphaNet：输入 + 特征提取层 + 池化层 + 特征展平/残差连接 + 降维度层 + 输出层
    '''

    def __init__(self, d=10, stride=10, d_pool=3, s_pool=3, n=9):
        super(AlphaNet, self).__init__()

        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride
        
        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride),
            nn.AvgPool1d(self.d, self.stride),
        ])

        # 特征提取层后面接的批归一化层
        self.batch_norms1 = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])

        # 池化层（无参数）
        self.avg_pool = nn.AvgPool1d(d_pool, s_pool)
        self.max_pool = nn.MaxPool1d(d_pool, s_pool)

        # 池化层后面接的批量归一化层
        self.batch_norms2 = nn.ModuleList([])
        for _ in range(2):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(h))
        for _ in range(5):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(n))

        # 特征展平并拼接后的总数
        n_in = 2 * (h * 2 * 3 + n * 5 * 3)

        # 线性层，输出层，激活函数，失活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear_layer = nn.Linear(n_in, 30)
        self.output_layer = nn.Linear(30, 1)



    def forward(self, X):

        features_fe, features_p, i = [], [], 0
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms1):
            # 特征提取 + 批量归一化 + 展平
            x = extractor(X)
            x = batch_norm(x)
            features_fe.append(x.flatten(start_dim=1))

            # 池化层 + 批量归一化 + 展平
            x_avg = self.batch_norms2[i](self.avg_pool(x))
            x_max = self.batch_norms2[i + 1](self.max_pool(x))
            x_min = self.batch_norms2[i + 2](-self.max_pool(-x))
            features_p.append(x_avg.flatten(start_dim=1))
            features_p.append(x_max.flatten(start_dim=1))
            features_p.append(x_min.flatten(start_dim=1))
            i += 3

        # 残差连接
        f1 = torch.cat(features_fe, dim=1)
        f2 = torch.cat(features_p, dim=1)
        features = torch.cat([f1, f2], dim=1)

        # 线性层 + 激活 + 失活 + 输出层
        features = self.linear_layer(features)
        features = self.relu(features)
        features = self.dropout(features)
        output = self.output_layer(features)

        return output

