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

# 数据库操作
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient


# ===================配置数据===================
# 读取yaml文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = ConfigBox(yaml.safe_load(f))


# 获取环境变量
MONGO_U = os.getenv('MONGO_U')
MONGO_P = os.getenv('MONGO_P')
MONGO_HOST = os.getenv('MONGO_HOST')
MONGO_PORT = os.getenv('MONGO_PORT')
MONGO_DB = os.getenv('MONGO_DB')


# 连接数据库
# uri 格式 = mongodb://用户名:密码@主机:端口
MONGO_URI = f"mongodb://{MONGO_U}:{MONGO_P}@{MONGO_HOST}:{MONGO_PORT}"
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB] # 数据库名称
# =============================================

# 模拟ETL过程 多数据源 写入 数据库
# ori_data -> db
async def ETL_to_db(pd_data:pd.DataFrame):
    try:
        # 数据处理
        pd_data.reset_index(drop=True, inplace=True) # 重置索引 确保索引从0开始
        dict_data = pd_data.to_dict('records') # pd.DataFrame -> dict

        # 按日期建表 如 collection_2025_12_25
        collection_name = f'collection_{today}'

        # 写入数据库
        await db[collection_name].delete_many({}) # 插入前清空此表 避免数据乱写入 保证数据一致性
        res = await db[collection_name].insert_many(dict_data) # 插入到db[这个表]中 批量插入

        # 返回结果
        logger.info(f"插入数据 库:{MONGO_DB} 表:{collection_name} 条数:{len(res.inserted_ids)}")
        return(len(res.inserted_ids))

    except Exception as e:
        raise LogError(e, sys)

# 被调用不会执行
if __name__ == "__main__":
    data = pd.read_csv(config.ori_data_path) # 读取原始数据
    asyncio.run(ETL_to_db(data))