import numpy as np
import pandas as pd
from typing import (
    List,
    Optional,
    Tuple
)

from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

from DataLoader import Kline


def data_standardize(df:pd.DataFrame,feature:str,method:str='quantile')->np.array:  # quantile or zscore

    scaler = StandardScaler()
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    pivot_df = df.pivot_table(index='trade_date', columns='st_code', values=feature)

    if method == 'quantile':
        standardized_array = np.stack(pivot_df.apply(lambda x:qt.fit_transform(x.values.reshape(-1,1)).reshape(-1), axis=1).values)
        standardized_df =pd.DataFrame(standardized_array, columns=pivot_df.columns,index=pivot_df.index)
    else:

        standardized_array =  np.stack(pivot_df.apply(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).reshape(-1),axis=1).values)
        standardized_df =pd.DataFrame(standardized_array, columns=pivot_df.columns,index=pivot_df.index)

    return standardized_df.unstack().dropna()

def data_transform(origin_data: pd.DataFrame, lookback:int=30) -> Tuple[np.array,np.array,np.array]:
    # 预先获取所有唯一的股票代码
    unique_codes = origin_data.index.map(lambda x:x[0]).unique()
    
    # 预分配内存
    X_list = []
    Y_list = []
    indices_list = []
    empty = []
    
    # 使用pandas的groupby操作，避免重复的loc操作
    grouped_data = origin_data.groupby(level=0)
    
    for code in tqdm(unique_codes):
        try:
            # 获取单个股票的数据
            df = grouped_data.get_group(code)
            
            # 使用numpy的滑动窗口操作
            data = df.iloc[:, :-1].values  # 特征数据
            returns = df['rtn'].values     # 收益率数据
            dates = df.index.get_level_values(1).values  # 日期数据
            
            # 计算可以生成的样本数量
            n_samples = len(df) - lookback + 1
            if n_samples <= 0:
                empty.append(code)
                continue
                
            # 使用numpy的滑动窗口创建特征矩阵
            X = np.lib.stride_tricks.sliding_window_view(data, (lookback, data.shape[1]))

            X = X[::5]
            X = X.reshape([X.shape[0],X.shape[2],X.shape[3]])  # 每5个交易日采样一次
            
            # 获取对应的标签和日期
            Y = returns[lookback-1::5]
            dates = dates[lookback-1::5]
            
            if len(X) == 0:
                empty.append(code)
                continue
                
            # 转置特征矩阵以匹配原始格式
            X = np.transpose(X, (0, 2, 1))
            
            # 创建索引
            indices = np.column_stack((np.full(len(X), code), dates))
            
            X_list.append(X)
            Y_list.append(Y)
            indices_list.append(indices)
            
        except Exception as e:
            empty.append(code)
            continue
    
    # 合并所有数据
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    
    print('Shape of X: ', X.shape)
    print('Shape of Y: ', Y.shape)
    print('Shape of indices: ', indices.shape)
    print('Stocks with not enough data: ', empty)
    
    return X, Y, indices

if __name__ == '__main__':
    my_data = Kline(start='20200101', end='20250201')
    data = my_data.Data
    data = data.sort_values(by=['st_code', 'trade_date'])
    data['rtn'] = data.groupby(by='st_code').apply(lambda x: (x.set_index('trade_date')['close_adj'].pct_change(5).shift(-5))).values  # predict the future 5 days
    data.loc[:, 'trade_state'], data.loc[:, 'close_adj'] = data.loc[:, 'close_adj'], data.loc[:, 'trade_state']
    data = data.rename(columns={'close_adj': 'trade_state', 'trade_state': 'close_adj'})
    data = data.dropna(axis=0)
    standard_sr_list = []
    for col in tqdm(data.columns[3:-1]):
        standard_feature = data_standardize(data,feature=col,method='zscore')
        standard_sr_list.append(standard_feature)
    standard_sr_list.append(data.set_index(['st_code','trade_date'])['rtn'])
    standard_data = pd.concat(standard_sr_list,axis=1)
    standard_data.columns = data.columns[3:]

    a,b,c = data_transform(standard_data)
    np.save('Features.npy', a)
    np.save('Labels.npy', b)
    np.save('indices.npy', c)
