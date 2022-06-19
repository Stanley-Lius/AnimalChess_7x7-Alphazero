# ====================
# 評估模組
# ====================

# 匯入套件
from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np

# 設定參數
EN_GAME_COUNT = 10 # 評估過程要進行的對戰次數（原始為 400）
EN_TEMPERATURE = 1.0  # 波茲曼分布的溫度參數

# 先手分數
def first_player_point(ended_state):
    # 1：先手獲勝、0：先手落敗、0.5：平手
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 進行 1 次完整對戰
def play(next_actions):
    state = State()

    # 循環直到遊戲結束
    while True:
        # 遊戲結束時
        if state.is_done():
            break;

        # 取得先手玩家與後手玩家的動作
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 取得下一個局勢 (盤面)
        state = state.next(action)

    # 傳回先手分數
    return first_player_point(state)

# 替換最佳玩家
def update_best_player():
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')

# 建構評估模組
def evaluate_network():
    # Step01 載入最新玩家與最佳玩家
    model0 = load_model('./model/latest.h5')
    model1 = load_model('./model/best.h5')

    # 建立利用 PV MCTS 選擇動作的函式
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # Step02 反覆進行數次對戰
    total_point = 0
    for i in range(EN_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 輸出狀態
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)

    # Step03 清除 session 與刪除模型
    K.clear_session()
    del model0
    del model1

    # Step04 勝率大於 0.5 時，就將最新玩家替換最佳玩家
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

# 單獨測試
if __name__ == '__main__':
    evaluate_network()
