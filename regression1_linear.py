import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt("click.csv", delimiter = ",", skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

#プロット
plt.plot(train_x, train_y, "o")
plt.show()