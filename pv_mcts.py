# ====================
# 策略價值蒙地卡羅樹搜尋法
# ====================

# 匯入套件
from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

# 準備參數
PV_EVALUATE_COUNT = 50

# 預測
def predict(model, state):
    # Step01 重塑訓練資料的 shape 以進行 model.predict( )
    a, b, c = DN_INPUT_SHAPE
    x = np.array(state.pieces_array())
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    # Step02 利用模型的預測去取得「策略」與「局勢價值」
    y = model.predict(x, batch_size=1)

    # Step03 取得策略
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1

    # Step04 取得局勢價值
    value = y[1][0][0]
    return policies, value

# 從每個節點中取出試驗次數
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# 利用 PV MCTS 取得試驗次數最多的棋步
def pv_mcts_scores(model, state, temperature):
    # Step01 定義 PV MCTS 的節點
    class Node:
        # 初始化節點
        def __init__(self, state, p):
            self.state = state # 盤面狀態
            self.p = p # 策略
            self.w = 0 # 累計價值
            self.n = 0 # 試驗次數
            self.child_nodes = None # 子節點群

        # 計算試驗次數
        def evaluate(self):
            # 情況 1 - 遊戲結束時
            if self.state.is_done():
                # 根據遊戲結果的勝敗取得局勢價值
                value = -1 if self.state.is_lose() else 0

                self.w += value
                self.n += 1
                return value

            # 情況 2 - 當子節點不存在時
            if not self.child_nodes:
                policies, value = predict(model, self.state)

                self.w += value
                self.n += 1

                # 擴充子節點
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value

            # 情況 3 - 當子節點存在時
            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value

        # 取得 PUCT 最大的子節點
        def next_child_node(self):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            return self.child_nodes[np.argmax(pucb_values)]

    # Step02 建構當前局勢的節點
    root_node = Node(state, 0)

    # Step03 執行多次模擬
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # Step04 取得合法棋步的機率分佈
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores

# 利用蒙地卡羅樹搜尋法選擇動作
def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# 波茲曼分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# 單獨測試
if __name__ == '__main__':
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))

    state = State()

    next_action = pv_mcts_action(model, 1.0)

    while True:
        if state.is_done():
            break

        action = next_action(state)

        state = state.next(action)

        print(state)