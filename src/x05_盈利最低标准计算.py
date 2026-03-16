import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 1. 固定业务参数
# ======================
# 测试给定范围内的 最低 xy 
a = 0.1   # 预测成本
b = 1.0   # 执行成本
c = 10.0   # 执行成功收益

# ======================
# 2. 构造 (x, y) 网格
# ======================
x = np.linspace(0.001, 1.0, 400)   # 覆盖率
y = np.linspace(0.001, 1.0, 400)   # 命中率
X, Y = np.meshgrid(x, y)

# ======================
# 3. 利润函数
# profit = x*y*c - a - x*b
# ======================
profit = X * Y * c - a - X * b

# ======================
# 4. 画 ROI 高原图
# ======================
plt.figure(figsize=(8, 6))

# 不盈利区域（蓝色）
plt.contourf(
    X, Y, profit,
    levels=[profit.min(), 0],
    alpha=0.7
)

# 盈利区域（红色，越深越赚钱）
plt.contourf(
    X, Y, profit,
    levels=30,
    alpha=0.9
)

plt.colorbar(label="Profit")
plt.xlabel("x (覆盖率)")
plt.ylabel("y (命中率 / Precision)")
plt.title(f"ROI 参数高原图  a={a}, b={b}, c={c}")

plt.show()





''' 管道化
重构为 Pipeline:
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])











from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 创建一个管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # 处理缺失值
    ('scaler', StandardScaler()),                        # 特征缩放
    ('classifier', LogisticRegression())                 # 分类器
])

# 网格搜索针对管道进行
params = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear']
}

grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1)
grid.fit(x_train, y_train)

# 这样保存的是 best_estimator_，它包含了【填充+缩放+模型】的所有参数
joblib.dump(grid.best_estimator_, 'model_v1.pkl')


'''