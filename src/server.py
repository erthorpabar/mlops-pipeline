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

# fastapi
import uvicorn
from fastapi import FastAPI

# 格式
from pydantic import BaseModel

# 命令行
import subprocess

# 模型
import pickle
import joblib

from typing import List, Dict
# ===================配置数据===================
# 读取yaml文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = ConfigBox(yaml.safe_load(f))

from pydantic_settings import BaseSettings # 优先系统环境变量，然后是.env文件，最后是默认值
class Settings(BaseSettings):
    
    # aws
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_DEFAULT_REGION: str = ""
    AWS_S3_NAME: str = ""

    # comfyui 配置
    COMFYUI_API_URL: str = " "

    # 数据库配置
    MYSQL1: str = " "
    MYSQL2: str = " "
    
    class Config:
        # 指定从.env文件加载环境变量
        env_file = ".env" # 允许从.env文件加载配置
        env_file_encoding = "utf-8" # 指定编码
        extra = "allow" # 允许额外的没用到的配置
        case_sensitive = True  # 环境变量大小写敏感

# 创建Settings的实例
# 在其他文件中，你可以通过导入settings来访问这些配置
settings = Settings()

# aws
AWS_ACCESS_KEY_ID = settings.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = settings.AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION = settings.AWS_DEFAULT_REGION
AWS_S3_NAME = settings.AWS_S3_NAME



# ==============公共变量================
v_list = None
model = None
using_model_version = None


from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    global v_list, model, using_model_version
    
    # 自动检测版本
    v_list = set() # 存储所有可以使用的模型版本
    os.makedirs(config.path.local_path, exist_ok=True) # 确保文件夹存在
    for i in os.listdir(config.path.local_path) : # 获取所有子文件夹名称 ['A_123', 'A_456', 'A_789']
        # 去除前缀
        version = i.replace('Artifact_', '')
        v_list.add(version)

    # 如果版本不为空
    # 加载模型版本
    if v_list:
        use_version = max(v_list) # 自动获取最大值
        local_model_path = config.path.local_model_path.format(today=use_version)
        # 如果文件夹存在 则加载模型
        if os.path.exists(local_model_path):
            # 加载
            model = joblib.load(local_model_path)
            logger.info(f"初始化加载模型 {local_model_path}")
            using_model_version = max(v_list)
    
    yield

app = FastAPI(lifespan=lifespan)

# 1 健康检查
@app.get("/health")
async def root():
    return {"message": "ok"}

# 2 获取版本列表 和当前使用的模型版本
@app.get("/version_list")
async def root():
    return {"version_list": v_list,"using_model_version": using_model_version}








# 3 下载最新模型并更新到本地
class Input(BaseModel):
    version_str: str
@app.post("/up_date_model")
async def root(request: Input):
    try:
        today = request.version_str
        logger.info(f"开始 下载到本地 {today}")

        # 该下载到哪个文件夹 a03_download/Artifact_{today}
        folder = config.path.download_path.format(today=today)
        os.makedirs(folder, exist_ok=True)

        # s3 url
        # s3://a1-s3-7765432/2025_12_27
        s3_url = f"s3://{AWS_S3_NAME}/Artifact_{today}"

        # 下载: S3 → 本地  
        # 同名下载会被覆盖
        # aws s3 cp s3://a1-s3-7765432/2025_12_27 a03_download/2025_12_27 --recursive
        command = f"aws s3 cp {s3_url} {folder} --recursive"
        
        # 创建副本 修改环境变量 这样不会修改全局环境变量
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
        env['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
        env['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION

        # 执行命令
        subprocess.run(command, shell=True, env=env)

        # log
        logger.info(f"完成 下载到本地 {folder}")
        v_list.add(today)
        return {"message": "下载成功"}
    except Exception as e:
        LogError(e, sys)
        return {"message": "下载失败"}


# 4 更换模型
class ChangeModelInput(BaseModel):
    version_str: str
@app.post("/change_model")
async def root(request: ChangeModelInput):
    try:
        today = request.version_str
        if today not in v_list:
            return {"message": "版本不存在"}
        else:
            logger.info(f"开始 更换模型 {today}")
            global using_model_version
            
            local_model_path = config.path.local_model_path.format(today=today)  # 
            model = joblib.load(local_model_path) 
            using_model_version = today  # 
            return {"message": "更换模型成功"}
    except Exception as e:
        LogError(e, sys)
        return {"message": "更换模型失败"}




# 5 单个推理
# 推理用的数据模型
class PredictInput(BaseModel):
    """推理输入数据模型"""
    having_IP_Address: int
    URL_Length: int
    Shortining_Service: int
    having_At_Symbol: int
    double_slash_redirecting: int
    Prefix_Suffix: int
    having_Sub_Domain: int
    SSLfinal_State: int
    Domain_registeration_length: int
    Favicon: int
    port: int
    HTTPS_token: int
    Request_URL: int
    URL_of_Anchor: int
    Links_in_tags: int
    SFH: int
    Submitting_to_email: int
    Abnormal_URL: int
    Redirect: int
    on_mouseover: int
    RightClick: int
    popUpWidnow: int
    Iframe: int
    age_of_domain: int
    DNSRecord: int
    web_traffic: int
    Page_Rank: int
    Google_Index: int
    Links_pointing_to_page: int
    Statistical_report: int

class PredictOutput(BaseModel):
    """推理输出数据模型"""
    type: int  # 0=正常网站, 1=钓鱼网站
    proba: float  # 预测为钓鱼网站的概率

# 单次推理
@app.post("/predict")
async def root(request: PredictInput) -> PredictOutput:
    try:
        # 没有处理缺失值
        # 转换为DataFrame
        input_df = pd.DataFrame([request.dict()])

        # 预测
        predict = model.predict_proba(input_df)
        p_proba = predict[0][1]  # 钓鱼网站的概率

        # 以0.5为阈值进行分类
        threshold = 0.5
        type = 1 if p_proba >= threshold else 0

        return PredictOutput(type=type, proba=p_proba)
    except Exception as e:
        LogError(e, sys)
        return PredictOutput(type=0, proba=0.111)


# 6 批量推理
# 批量推理验证效果
from fastapi import File, UploadFile
@app.post("/predict_batch")
async def root(file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
         # 移除目标变量列（如果存在）
        if 'Result' in df.columns:
            df_features = df.drop('Result', axis=1)
        else:
            df_features = df
        # 预测
        predict = model.predict_proba(df_features)
        p_proba = predict[:, 1]  # 钓鱼网站的概率

        threshold = 0.5
        type = (p_proba >= threshold).astype(int)

        results = []
        for i in range(len(type)):
            results.append({
                "type": int(type[i]),
                "proba": float(p_proba[i])
            })
        
        return {"results": results}

       
    except Exception as e:
        LogError(e, sys)
        return {"message": "预测失败"}

# ——————————————启动服务——————————————
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7004)) # 端口
    host = os.getenv("HOST", "0.0.0.0") # 主机

    uvicorn.run("server:app", host=host, port=port,reload=False) # 启动服务

