# 时间格式 2025_12_25
import time
today = time.strftime("%Y_%m_%d")

# 日志
import os      
import sys     
import logging 




# [时间: 级别: 模块: 消息]
# 示例输出 = [2025-12-23 10:30:45,123: INFO: server: 服务器启动成功]
log_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# 每天一个log文件
# 日志文件名 = 2025_12_24.log
log_name = f"{today}.log"

# 创建文件夹
log_dir = os.path.join(os.path.dirname(__file__), 'logs') # 当前文件夹/logs
os.makedirs(log_dir, exist_ok=True) # 确保文件夹存在

# log保存路径
log_path = os.path.join(log_dir, log_name) # 当前文件夹/logs/2025_12_24.log

# 配置
logging.basicConfig(
    level=logging.INFO, # 记录级别 = logging.INFO
    format=log_format, # 格式 = log_format
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'), # 日志写入地址 = log_file_path
        logging.StreamHandler(sys.stdout) # 日志输出到控制台 = sys.stdout
    ]
)

# 标识符 = ml_pipeline_logger
logger = logging.getLogger("ml_pipeline_logger")

# 被调用不会执行
if __name__ == "__main__":
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.debug("这是一条调试日志")  # 注意：由于 level=logging.INFO，DEBUG 级别不会显示
