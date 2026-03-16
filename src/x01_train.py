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
os.environ["NUMEXPR_MAX_THREADS"] = "8"

# 数据库操作
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# 数据格式
from pydantic import BaseModel, Field
from typing import Literal

# 切分 训练集 测试集
from sklearn.model_selection import train_test_split
# KS检验：用于检测两个样本是否来自同一分布（数据漂移检测）
from scipy.stats import ks_2samp
# knn临近算法填充缺失值
from sklearn.impute import KNNImputer

# 模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# 网格搜索
from sklearn.model_selection import GridSearchCV

# 评估
from sklearn import metrics

# 作图
import matplotlib.pyplot as plt

# 保存模型
import joblib

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


# 数据模型
class DataSchema(BaseModel):
    """网络钓鱼检测数据模型"""
    
    having_IP_Address: Literal[-1, 0, 1] = Field(description="IP地址特征")
    URL_Length: Literal[-1, 0, 1] = Field(description="URL长度特征")
    Shortining_Service: Literal[-1, 1] = Field(description="短链接服务")
    having_At_Symbol: Literal[-1, 1] = Field(description="包含@符号")
    double_slash_redirecting: Literal[-1, 1] = Field(description="双斜杠重定向")
    Prefix_Suffix: Literal[-1, 1] = Field(description="前缀后缀")
    having_Sub_Domain: Literal[-1, 0, 1] = Field(description="子域名")
    SSLfinal_State: Literal[-1, 0, 1] = Field(description="SSL状态")
    Domain_registeration_length: Literal[-1, 0, 1] = Field(description="域名注册长度")
    Favicon: Literal[-1, 1] = Field(description="网站图标")
    port: Literal[-1, 1] = Field(description="端口")
    HTTPS_token: Literal[-1, 1] = Field(description="HTTPS令牌")
    Request_URL: Literal[-1, 0, 1] = Field(description="请求URL")
    URL_of_Anchor: Literal[-1, 0, 1] = Field(description="锚点URL")
    Links_in_tags: Literal[-1, 0, 1] = Field(description="标签中的链接")
    SFH: Literal[-1, 0, 1] = Field(description="服务器表单处理")
    Submitting_to_email: Literal[-1, 1] = Field(description="提交到邮箱")
    Abnormal_URL: Literal[-1, 1] = Field(description="异常URL")
    Redirect: Literal[0, 1] = Field(description="重定向")
    on_mouseover: Literal[-1, 0, 1] = Field(description="鼠标悬停")
    RightClick: Literal[-1, 0, 1] = Field(description="右键点击")
    popUpWidnow: Literal[-1, 0, 1] = Field(description="弹出窗口")
    Iframe: Literal[-1, 0, 1] = Field(description="iframe标签")
    age_of_domain: Literal[-1, 0, 1] = Field(description="域名年龄")
    DNSRecord: Literal[-1, 0, 1] = Field(description="DNS记录")
    web_traffic: Literal[-1, 0, 1] = Field(description="网络流量")
    Page_Rank: Literal[-1, 0, 1] = Field(description="页面排名")
    Google_Index: Literal[-1, 0, 1] = Field(description="Google索引")
    Links_pointing_to_page: Literal[-1, 0, 1] = Field(description="指向页面的链接")
    Statistical_report: Literal[-1, 0, 1] = Field(description="统计报告")
    Result: Literal[-1, 1] = Field(description="目标变量: -1=钓鱼网站, 1=正常网站")


async def train(today:str):
    try:
        # === 1 加载数据 
        logger.info(f"—————— 加载原始数据 {today} ——————")

        collection_name = f'collection_{today}' # 表名
        data = pd.DataFrame(await db[collection_name].find().to_list(length=None)) # 加载数据 -> np.dataframe
        if data.empty:
            raise ValueError(f"表 {collection_name} 为空 今天没有数据 {today}")
        
        logger.info(f"数据 长度 {len(data)} 条数据")

        if '_id' in data.columns: # 删除_id列 (MongoDB自动生成的字段)
            data.drop(columns=['_id'], inplace=True)
        
        # === 2 通用数据预处理
        # 1 字段完整性
        schema_set = set(DataSchema.model_fields.keys()) # 模型字段
        data_set = set(data.columns) # 数据字段

        # 检查多余字段
        extra_x = data_set - schema_set
        if extra_x:
            raise ValueError(f'数据 中有多余字段{extra_x}')

        # 检查缺失字段
        missing_x = schema_set - data_set
        if missing_x:
            raise ValueError(f'模型 中有多余字段{missing_x}')

        # 检查完成
        logger.info(f'字段完整 无多余字段 无缺失字段 包含 {len(schema_set)} 个字段')

        # === 3 缺失值处理
        data.dropna(inplace=True) # 删除包含缺失值的行
        data.replace({"na": np.nan}, inplace=True) # 将 "na" 替换为 np.nan 便于后续处理

        logger.info(f'删除缺失值 后 剩余 {len(data)} 条数据')

        # === 4 保存训练数据
        data_path = config.path.data_path.format(today=today)
        os.makedirs(os.path.dirname(data_path), exist_ok=True) # 确保父目录存在
        data.to_csv(data_path, index=False, header=True) # 保存 不含索引 包含列名称
        logger.info(f"保存训练数据 路径:{data_path} 条数:{len(data)}")


        # === 5 切分 x y ===
        x = data.drop(['Result'],axis = 1) # 排除 预测目标列
        y = data[['Result']] # 拿到 预测目标列([target]-> np.series 或 [[target]]-> np.dataframe)

        # === 6 切分 训练集 和 测试集 ===
        x_train,x_test,y_train,y_test=train_test_split(
            x, # 特征数据集
            y, # 标签数据集
            test_size=0.2, # 随机0.2作为测试
            random_state=0 # 随机拆分种子
        )

        # === 7 输入前处理 ===
        # 1 对连续型数值 进行 归一化

        # 2 填充缺失值
        ''' 
        from sklearn.impute import KNNImputer

        d = {
            "missing_values": np.nan,   # 要填充的缺失值类型(NaN)
            "n_neighbors": 3,           # 使用3个最近邻居
            "weights": "uniform",       # 邻居权重相等（也可选"distance"按距离加权）
        }

        x = pd.DataFrame(imputer.fit_transform(x),columns=x.columns, index=x.index)
        # inputer 模型要保存 - 在推理的时候要用
        '''

        # 3 对str类型的字段 进行 独热编码
        # x = pd.get_dummies(x,drop_first=True,dtype = int) 
        ''' 
        get_dummies 独热编码 将分类变量转换为 bool 或 int
        get_dummies 的输出为多列0/1的组合

        dtype = int 将分类变量转换为 int 第一个=00 第二个= 01
        dtype = bool 将分类变量转换为 bool

        drop_first=True
        原始->  B  C
        A      0  0
        B      1  0
        C      0  1

        drop_first=false
        原始->  A  B  C
        A      1  0  0
        B      0  1  0
        C      0  0  1
        '''

        # 4 对标签 y 进行调整(手动)
        # 将 [-1,1] 映射成 [0,1]
        y_train = y_train.replace(-1, 0)
        y_test = y_test.replace(-1, 0)

        # 5 对标签 y 进行调整(标签编码器)
        # 将 [-1,1] 映射成 [0,1]
        # from sklearn.preprocessing import LabelEncoder
        # labelencoder = LabelEncoder()
        # y = labelencoder.fit_transform(y['Result'].values)
        ''' 
        需要保存 labelencoder 模型
        在推理完成后
        需要调用 labelencoder.inverse_transform() 将 0 1 映射回 [-1,1]
        才能让业务系统正确识别
        '''        

        # === 8 训练 ===
        # 1 逻辑回归
        Logistic = LogisticRegression(verbose=1)
        params_Logistic = {
            'C': [1],
            'penalty': ['l2'],
            'solver': ['liblinear'],
        }
        grid_Logistic = GridSearchCV(Logistic, params_Logistic, cv=3,scoring="roc_auc")
        grid_Logistic.fit(x_train, y_train)
        best_Logistic = grid_Logistic.best_estimator_

        # 2 决策树
        DecisionTree = DecisionTreeClassifier()
        params_DecisionTree = {
            'criterion': ['gini'],
            'max_leaf_nodes': [10], 
            'max_depth': [5],
        }
        grid_DecisionTree = GridSearchCV(DecisionTree, params_DecisionTree, cv=3,scoring="roc_auc")
        grid_DecisionTree.fit(x_train, y_train)
        best_DecisionTree = grid_DecisionTree.best_estimator_

        # 3 bagging
        Bagging = BaggingClassifier(n_estimators=100, random_state=0)
        params_Bagging = {
            'n_estimators': [50],
            'max_samples': [0.8],
            'max_features': [0.8],
        }
        grid_Bagging = GridSearchCV(Bagging, params_Bagging, cv=3,scoring="roc_auc")
        grid_Bagging.fit(x_train, y_train)
        best_Bagging = grid_Bagging.best_estimator_

        # 4 随机森林
        RandomForest = RandomForestClassifier(verbose=1) 
        params_RandomForest = {
            'n_estimators': [50],
            'max_depth': [10],
        }
        grid_RandomForest = GridSearchCV(RandomForest, params_RandomForest, cv=3,scoring="roc_auc")
        grid_RandomForest.fit(x_train, y_train)
        best_RandomForest = grid_RandomForest.best_estimator_

        # 5 boosting
        GradientBoosting = GradientBoostingClassifier(verbose=1)
        params_GradientBoosting = {
            'learning_rate': [0.1],
            'subsample': [0.8],
            'n_estimators': [50],
        }
        grid_GradientBoosting = GridSearchCV(GradientBoosting, params_GradientBoosting, cv=3,scoring="roc_auc")
        grid_GradientBoosting.fit(x_train, y_train)
        best_GradientBoosting = grid_GradientBoosting.best_estimator_

        # 6 adaboost
        AdaBoost = AdaBoostClassifier()
        params_AdaBoost = {
            'learning_rate': [0.1],
            'n_estimators': [50],
        }
        grid_AdaBoost = GridSearchCV(AdaBoost, params_AdaBoost, cv=3,scoring="roc_auc")
        grid_AdaBoost.fit(x_train, y_train)
        best_AdaBoost = grid_AdaBoost.best_estimator_

        # === 9 评估 计算指标 ===
        # 1 预测结果 概率形式
        y_pred_Logistic = best_Logistic.predict_proba(x_test)[:,1]
        y_pred_DecisionTree = best_DecisionTree.predict_proba(x_test)[:,1]
        y_pred_Bagging = best_Bagging.predict_proba(x_test)[:,1]
        y_pred_RandomForest = best_RandomForest.predict_proba(x_test)[:,1]
        y_pred_GradientBoosting = best_GradientBoosting.predict_proba(x_test)[:,1]
        y_pred_AdaBoost = best_AdaBoost.predict_proba(x_test)[:,1]


        # 2 作图 roc curvue
        fpr1 , tpr1, thresholds1 = metrics.roc_curve(y_test, y_pred_Logistic)
        auc1 = metrics.roc_auc_score(y_test, y_pred_Logistic) # auc值

        fpr2 , tpr2, thresholds2 = metrics.roc_curve(y_test, y_pred_DecisionTree)
        auc2 = metrics.roc_auc_score(y_test, y_pred_DecisionTree) # auc值

        fpr3 , tpr3, thresholds3 = metrics.roc_curve(y_test, y_pred_Bagging)
        auc3 = metrics.roc_auc_score(y_test, y_pred_Bagging) # auc值

        fpr4 , tpr4, thresholds4 = metrics.roc_curve(y_test, y_pred_RandomForest)
        auc4 = metrics.roc_auc_score(y_test, y_pred_RandomForest) # auc值

        fpr5 , tpr5, thresholds5 = metrics.roc_curve(y_test, y_pred_GradientBoosting)
        auc5 = metrics.roc_auc_score(y_test, y_pred_GradientBoosting) # auc值

        fpr6 , tpr6, thresholds6 = metrics.roc_curve(y_test, y_pred_AdaBoost)
        auc6 = metrics.roc_auc_score(y_test, y_pred_AdaBoost) # auc值

        # 设置中文字体显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        plt.plot([0,1],[0,1], 'k--')
        plt.plot(fpr1, tpr1, label= f"LogisticRegression auc={auc1:.4f}")
        plt.plot(fpr2, tpr2, label= f"DecisionTree auc={auc2:.4f}")
        plt.plot(fpr3, tpr3, label= f"Bagging auc={auc3:.4f}")
        plt.plot(fpr4, tpr4, label= f"RandomForest auc={auc4:.4f}")
        plt.plot(fpr5, tpr5, label= f"GradientBoosting auc={auc5:.4f}")
        plt.plot(fpr6, tpr6, label= f"AdaBoost auc={auc6:.4f}")

        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f'ROC Curve 日期:{today}')

        # 保存 roc 曲线图片
        roc_curve_path = config.path.ROC_curve_path.format(today=today)
        os.makedirs(os.path.dirname(roc_curve_path), exist_ok=True)
        plt.savefig(roc_curve_path)
        plt.close()
        logger.info(f"ROC曲线图片 保存路径:{roc_curve_path}")


        # 3 计算其他指标 
        ''' 
        ====== 商业价值估算 ======
        以广告投放为例
        展示广告成本 = 1元/次
        点击预期收益 = 10元/次
        预测成本 = 0.1元/次

        ====== 混淆矩阵 ======
        TP                              | FP
        预估 会买                        | 预估 会买
        实际 会买                        | 实际 不会买
        营收 = 10元                       | 营收 = 0元     
        成本 = 1.1元                        | 成本 = 1.1元
        ————————————————————————————————————————————————————
        FN                              | TN
        预估 不会买                      | 预估 不会买
        实际 会买                        | 实际 不会买
        营收 = -0元(10元潜在损失)        | 营收 = 0元
        成本 = 0.1元                     | 成本 = 0.1元


        ====== 计算roi =====
        假设共投放2000次广告

        a模型
        | TP(50)  | FP(100)  |  
        | FN(50)  | TN(1800)  |
        准确率 = (50 + 1800) / 2000 = 0.925 = 92.5%
        营收 = 10*TP = 500
        成本 = 1.1(TP+FP) + 0.1(FN+TN) = 1.1*150 + 0.1*1850 = 165 + 185 = 350
        利润 = 500 - 350 = 150
        roi = 150/350 = 42.86%
        
        b模型
        | TP(60)  | FP(50)  |  
        | FN(40)  | TN(1850)  |
        准确率 = (60 + 1850) / 2000 = 0.955 = 95.5%
        营收 = 10*TP = 600
        成本 = 1.1(TP+FP) + 0.1(FN+TN) = 1.1*90 + 0.1*1890 = 99 + 189 = 285
        利润 = 600 - 285 = 315
        roi = 315/285 = 110.53%

        总结
        准确率 提升 3% roi 提升 67.67%

        本质 
        1 预测成本 a
        2 执行成本 b
        3 执行成功收益 c
        4 成功概率 p -> 模型

        这四个因素公共决定了模型的商业价值
        
        '''

        models_list = [
            # 模型名称, 模型对象, 预测结果
            ('LogisticRegression', best_Logistic, y_pred_Logistic),
            ('DecisionTree', best_DecisionTree, y_pred_DecisionTree),
            ('Bagging', best_Bagging, y_pred_Bagging),
            ('RandomForest', best_RandomForest, y_pred_RandomForest),
            ('GradientBoosting', best_GradientBoosting, y_pred_GradientBoosting),
            ('AdaBoost', best_AdaBoost, y_pred_AdaBoost)
        ]

        res = []
        for model_type,model,y_pred_proba in models_list:
            # 以 0.5 为阈值 转换为 0 1 二分类
            u = 0.5 # 阈值
            y_pred = (y_pred_proba >= u).astype(int)

            # 混淆矩阵
            ''' 
            《《《 指标 - roi 映射关系 》》》

            ===== 1 混淆矩阵 =====
            TP           | FP
            预测 会买     | 预测 会买
            实际 会买     | 实际 不会买
            ————————————————————————————
            FN           | TN
            预测 不会买   | 预测 不会买
            实际 会买     | 实际 不会买

            ===== 2 成本和收益 =====
            预测成本 = a
            执行成本 = b
            执行成功收益 = c

            1 所有样本 -> 都会被预测 -> 会买/不会买      [成本 = (tp+fp+fn+tn) * a]
            2 会买 -> 触发执行 -> 成功/失败             [成本 (tp+fp) * b]
            3 成功 -> 创造收益                          [收益 = tp * c]

            成本 = (tp+fp+fn+tn) * a + (tp+fp) * b
            营收 = tp * c
            利润 = 营收 - 成本
            roi = 利润 / 成本

            ===== 3 指标理解 =====
            roi = (c-b-a)tp - (b+a)fp - fn - tn
            
            accuracy = 预测正确 / 预测总数 = (tp + tn) / (tp + tn + fp + fn)
            -> 预测的正确性
            -> 隐含假设 tp tn fp fn 价值相同 的情况下 越高越好

            precision = 实际成功 / 预测为成功 = tp / (tp + fp)
            -> 预测为成功 实际有多少比例真正成功了
            -> pre↑ -> [fp↓] -> 减少b中无效执行的浪费

            specificity = 实际不成功 / 本来有多少不成功 = tn / (tn + fp)
            -> 本来不成功 有多少比例被找出来了
            -> spe↑ -> [fp↓] -> 减少b中无效执行的浪费

            recall = 实际成功 / 本来有多少成功 = tp / (tp + fn)
            -> 本来能成功 有多少比例被找出来了
            -> recall↑ -> [fn↓](减少少赚的钱) -> 增加c的数量

            《《《 以上指标 只关注 混淆矩阵的比例 》》》

            最终指标还是要看roi


            '''
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

            # 准确率 = (真正例 + 真负例) / 总样本数
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # 精确率 = 真正例 / (真正例 + 假正例) - 预测为正的样本中有多少是真正的正例
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # 召回率 = 真正例 / (真正例 + 假负例) - 实际为正的样本中有多少被正确预测
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 特异度 = 真负例 / (真负例 + 假正例) - 实际为负的样本中有多少被正确预测
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # 特异度

            # F1分数 = 2 * 精确率 * 召回率 / (精确率 + 召回率) - 精确率和召回率的调和平均值
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 # f1分数

            # 马修斯相关系数(MCC)
            # MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) 
            denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0

            # AUC 曲线下面积 - 衡量分类器在不同阈值下的性能
            auc_score = metrics.roc_auc_score(y_test, y_pred_proba) # 这里用概率去计算

            # 商业价值
            a = config.roi.a # 预测成本
            b = config.roi.b # 执行成本
            c = config.roi.c # 执行成功收益
            
            revenue = tp * c # 营收
            cost = (tp + fp + fn + tn) * a + (tp + fp) * b # 成本
            profit = revenue - cost # 利润
            roi = profit / cost # ROI


            # 记录结果
            res.append({
                'model': model_type,

                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,

                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'mcc': mcc,
                'auc': auc_score,

                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'roi': roi,
            })

        # 4 保存为csv
        # list -> pd.DataFrame
        res = pd.DataFrame(res)
        # 将ROI格式化为百分比显示
        res['roi'] = res['roi'].apply(lambda x: f"{x * 100:.2f}%")
        # 保存 各指标的表格数据
        table_csv_path = config.path.table_csv_path.format(today=today)
        os.makedirs(os.path.dirname(table_csv_path), exist_ok=True)
        res.to_csv(table_csv_path, index=False, header=True)
        logger.info(f"结果 csv 保存路径:{table_csv_path}")


        # 5 保存为PDF报告 (使用matplotlib)
        table_image_path = config.path.table_image_path.format(today=today)
        os.makedirs(os.path.dirname(table_image_path), exist_ok=True)
        
        # 创建一个新的figure用于表格
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 准备表格数据
        models = res['model'].tolist()
        
        # 创建表格数据
        table_data = []
        
        # 表头
        table_data.append(['指标/Model'] + models)
        
        # 混淆矩阵
        table_data.append(['TN (真负例)'] + [int(row['tn']) for _, row in res.iterrows()])
        table_data.append(['FP (假正例)'] + [int(row['fp']) for _, row in res.iterrows()])
        table_data.append(['FN (假负例)'] + [int(row['fn']) for _, row in res.iterrows()])
        table_data.append(['TP (真正例)'] + [int(row['tp']) for _, row in res.iterrows()])
        
        # 空行
        table_data.append([''] * (len(models) + 1))
        
        # 评估指标
        table_data.append(['准确率'] + [f"{row['accuracy']:.4f}" for _, row in res.iterrows()])
        table_data.append(['精确率'] + [f"{row['precision']:.4f}" for _, row in res.iterrows()])
        table_data.append(['召回率'] + [f"{row['recall']:.4f}" for _, row in res.iterrows()])
        table_data.append(['特异性'] + [f"{row['specificity']:.4f}" for _, row in res.iterrows()])
        table_data.append(['F1分数'] + [f"{row['f1']:.4f}" for _, row in res.iterrows()])
        table_data.append(['MCC'] + [f"{row['mcc']:.4f}" for _, row in res.iterrows()])
        table_data.append(['AUC'] + [f"{row['auc']:.4f}" for _, row in res.iterrows()])
        
        # 空行
        table_data.append([''] * (len(models) + 1))
        
        # 商业价值
        table_data.append(['营收'] + [f"{row['revenue']:.0f}" for _, row in res.iterrows()])
        table_data.append(['成本'] + [f"{row['cost']:.1f}" for _, row in res.iterrows()])
        table_data.append(['利润'] + [f"{row['profit']:.1f}" for _, row in res.iterrows()])
        table_data.append(['ROI'] + [str(row['roi']) for _, row in res.iterrows()])
        
        # 创建表格
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置标题
        plt.title(f'机器学习模型评估结果对比表\n生成时间: {today.replace("_", "-")}', 
                 fontsize=14, pad=20)
    

        # 保存为PNG图片
        plt.savefig(table_image_path, bbox_inches='tight', dpi=300, facecolor='white')
        logger.info(f"结果 PNG图片 保存路径:{table_image_path}")

        




        # === 10 保存最佳模型 ===
        # 5 使用auc 指标 筛选出 最佳模型 并 保存模型
        best_result = max([
            ('LogisticRegression', best_Logistic, auc1),
            ('DecisionTree', best_DecisionTree, auc2),
            ('Bagging', best_Bagging, auc3),
            ('RandomForest', best_RandomForest, auc4),
            ('GradientBoosting', best_GradientBoosting, auc5),
            ('AdaBoost', best_AdaBoost, auc6)
        ], key=lambda x: x[2])  # 按第3个元素(AUC)比较
        # max() -> 使用 key 比较大小
        # key=lambda x: x[2] -> 比较第三个元素
        # best_result -> (模型名称, 模型对象, auc值)

        model_name, model_object, auc = best_result

        model_path = config.path.model_path.format(today=today)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_object, model_path)
        logger.info(f"最佳模型 保存路径:{model_path}")
            


    except Exception as e:
        raise LogError(e, sys)

# 被调用不会执行
if __name__ == "__main__":
    asyncio.run(train(today))

