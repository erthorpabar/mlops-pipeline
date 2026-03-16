import sys
from w01_log import logger

class LogError(Exception): # 继承 py 内置 Exception 类
    def __init__(self, error_message, error_details: sys):
        
        # 错误信息
        self.error_message = error_message

        # exc_info() 返回三个值: 异常类型、异常值、错误对象
        _,_,exc_tb = error_details.exc_info()  
        
        # 若有错误 即 except as e 的情况下 使用此函数
        if exc_tb:
            self.lineno = exc_tb.tb_lineno # 错误行号
            self.file_name = exc_tb.tb_frame.f_code.co_filename # 错误文件名

        # 若无错误 即 在 raise 的情况下 使用此函数
        else: 
            import traceback
            stack = traceback.extract_stack()[:-1] # 
            frame = stack[-1] if stack else None # 获取调用者位置信息
            self.lineno = frame.lineno if frame else -1 # 错误行号
            self.file_name = frame.filename if frame else "unknown" # 错误文件名

        # 写入日志记录
        logger.error(self)

    def __str__(self):
        return f'''
发生于 {self.file_name} 第 {self.lineno} 行
Error: {self.error_message} 
'''
        

# 被调用不会执行
if __name__ == "__main__":
    try:
        logger.info("开始计算") # 正常主动记录日志
        # raise RuntimeError("这是一个运行时错误")
        a = 1/0 # 这里会触发异常
    except Exception as e:
        raise LogError(e, sys) # 异常 并 自动记录日志

''' 
使用方法

try:
    logger.info("这是个日志")
    raise RuntimeError("这是个错误")
    # 这里不要写任何logError 因为所有报错 会在 except 中被捕获 并自动记录日志 否则会重复记录日志
except Exception as e:
    raise LogError(e, sys) 
'''