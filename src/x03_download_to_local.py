# 时间
import time
# 格式 2025_12_25
today = time.strftime("%Y_%m_%d")

# 加载环境变量
import os
from dotenv import load_dotenv
load_dotenv()

# log
import sys
from w01_log import logger
from w02_log_error import LogError

# yaml
import yaml
from box import ConfigBox

# 异步
import asyncio

# 执行系统命令
import subprocess 

# 获取环境变量
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_S3_NAME = os.getenv('AWS_S3_NAME')


# 读取yaml文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = ConfigBox(yaml.safe_load(f))

async def download_to_local():
    try:
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

    except Exception as e:
        raise LogError(e, sys)

if __name__ == "__main__":
    asyncio.run(download_to_local())