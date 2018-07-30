import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt("click.csv", delimiter=",", skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# 平均二乗誤差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum(y - f(x)) ** 2

# パラメータを初期化
theta = np.random.rand(3)

# 学習データの行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

# 予測関数
def f(x):
    return np.dot(x, theta)

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 標準化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

X = to_matrix(train_z)

# 平均二乗誤差の履歴
errors = []

# 学習率
ETA = 1e-3

# 誤差の差分
diffs = 1
diff = 1

# 学習を繰り返す
errors.append(MSE(X, train_y))
error = E(X, train_y)
while diff > 1e-2:
    # パラメータ更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    # 前回の誤差と差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

    diffs = errors[-2] - errors[-1]

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, "o")
plt.plot(x, f(to_matrix(x)))
plt.show()

# 誤差をプロット

x2 = np.arange(len(errors))

plt.plot(x2, errors)
plt.show()