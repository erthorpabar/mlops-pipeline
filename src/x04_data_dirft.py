# 时间格式 2025_12_25
import time
today = time.strftime("%Y_%m_%d")

# log
import sys
from w01_log import logger
from w02_log_error import LogError

# yaml
import yaml
from box import ConfigBox

# 加载环境变量
import os
from dotenv import load_dotenv
load_dotenv()

# 异步
import asyncio

# 数据处理
import pandas as pd
import numpy as np

# 分割
from sklearn.model_selection import train_test_split

# 分布一致性
from scipy.stats import ks_2samp

# ===================配置数据===================
# 读取yaml文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = ConfigBox(yaml.safe_load(f))

# =============================================
''' 
三种检测方式 当前实现1种
psi
kl divergence
ks

'''
# 加载csv
def data_drift():
    try:
        path = config.path.data_path.format(today=today)
        df = pd.read_csv(path)


        df_a,df_b=train_test_split(
            df, 
            test_size=0.2, # 随机0.2作为测试
            random_state=0 # 随机拆分种子
        )

        dist_same = True
        for c in df_a.columns:
            d1 = df_a[c]
            d2 = df_b[c]

            # 当前验证 train 和 test 数据集 是否因为 切分而导致分布不一致
            # 实际应该验证 旧数据 和 新数据 之间是否分布一致

            # 原理
            # 两列同类型 数据 a1 a2
            # 累积分布函数 CDF -> 
            # D: 两个CDF之间的最大距离 
            # P: 两个样本来自同一个分布的概率 
            # p>= 0.05 置信度95% 来自同一分布 
            # p< 0.05 显著差异 来自不同分布

            # 累积分布函数 CDF
            _ , pvalue = ks_2samp(d1,d2) # 返回 (距离,概率) statistic=0.02, pvalue=0.85
            if pvalue >= 0.05:
                pass
            else:
                dist_same = False
                logger.info(f"数据 字段 {c} 分布不一致 置信度: {pvalue}")
        
        # 如果通过了分布一致性
        if dist_same:
            logger.info(f"数据分布一致性验证通过: 所有 {len(df_a.columns)} 个字段的分布一致")

    except Exception as e:
        raise LogError(e, sys)

if __name__ == "__main__":
    data_drift()