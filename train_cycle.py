# ====================
# 訓練循環
# ====================

# 匯入套件
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network

# 建立對偶網路
dual_network()

for i in range(10):
    print('Train',i,'====================')
    # 自我對弈模組
    self_play()

    # 訓練模組
    train_network()

    # 評估模組
    evaluate_network()