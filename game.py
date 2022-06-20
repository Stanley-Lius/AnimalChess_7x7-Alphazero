# ====================
# 動物棋
# ====================

# 匯入套件
import random
import math

# 遊戲局勢
class State:
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        # 方向常數
        self.dxy = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

        # 傳入雙方的棋子作判斷
        self.pieces = pieces if pieces != None else [0] * (36+3)
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * (36+3)
        self.depth = depth

        # 棋子的初始配置 (在這裡才擺放動物)
        if pieces == None or enemy_pieces == None:
            self.pieces = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 3, 0, 0, 0, 0]
            self.enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 3, 0, 0, 0, 0]
          
    # 判斷是否落敗
    def is_lose(self):
        for i in range(49):
            if self.pieces[i] == 4: # 獅子存在
                return False
        return True

    # 判斷是否平手
    def is_draw(self):
        return self.depth >= 300 # 300手

    # 判斷遊戲是否已結束
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # 轉換動物棋的局勢
    def pieces_array(self):
        # 取得每位玩家的局勢 (盤面)
        def pieces_array_of(pieces):
            table_list = []
            #棋子 ID「1」：雞、棋子 ID「2」：象、棋子 ID「3」：鹿、棋子 ID「4」：獅
            for j in range(1, 5):
                table = [0] * 49
                table_list.append(table)
                for i in range(49):
                    if pieces[i] == j:
                        table[i] = 1

            # 棋子 ID「5」：雞持駒、棋子 ID「6」：象持駒、棋子 ID「7」：鹿持駒
            for j in range(1, 4):
                flag = 1 if pieces[48+j] > 0 else 0
                table = [flag] * 49
                table_list.append(table)
            return table_list

        # 組合成對偶網路的輸入的 (2 軸陣列)
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]

    # 將棋子移動目的地與移動的方向轉換成動作
    def position_to_action(self, position, direction):
        return position * 11 + direction
 
    # 將動作轉換成棋子移動目的地與起始地
    def action_to_position(self, action):
        return (int(action/11), action%11)

    # 取得合法棋步的串列
    def legal_actions(self):
        actions = []
        for p in range(49):
            # 移動棋子
            if self.pieces[p]  != 0:
                actions.extend(self.legal_actions_pos(p))

            # 放回持駒
            if self.pieces[p] == 0 and self.enemy_pieces[48-p] == 0:
              
                for capture in range(1,4):
                    
                    if self.pieces[48+capture] != 0:
                        actions.append(self.position_to_action(p, 8-1+capture))
 
        return actions

    # 取得移動棋子時的合法棋步串列
    def legal_actions_pos(self, position_src):
        actions = []

        # 先區分每個棋子移動的方向
        piece_type = self.pieces[position_src]
        if piece_type > 4: piece_type-4
        directions = []
        if piece_type == 1: # 小雞
            directions = [0]
        elif piece_type == 2: # 大象
            directions = [1, 3, 5, 7]
        elif piece_type == 3: # 長頸鹿
            directions = [0, 2, 4, 6]
        elif piece_type == 4: # 獅子
            directions = [0, 1, 2, 3, 4, 5, 6, 7]

        # 區分完每個棋子的方向後，利用公式取得合法棋步
        for direction in directions:
            # 利用棋子的起始地計算出 p，p 代表著棋盤的索引
            x = position_src%7 + self.dxy[direction][0]
            y = int(position_src/7) + self.dxy[direction][1]
            p = x + y * 7

            # 可移動時加入合法棋步
            if 0 <= x and x <= 6 and 0<= y and y <= 6 and self.pieces[p] == 0:
                actions.append(self.position_to_action(p, direction))
        return actions

    # 取得下一個局勢 (盤面)
    def next(self, action):
        # 透過複製上一個的局勢 (盤面) 建立下一個局勢 (盤面) 並把 depth 加上 1
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth+1)

        # 將動作透過 action_to_position(action) 轉換成（移動目的地, 移動起始地）
        position_dst, position_src = self.action_to_position(action)

        # 移動棋子
        if position_src < 8:
            # 利用棋子的目的地及方向常數計算棋子的移動起始地
            x = position_dst%7 - self.dxy[position_src][0]
            y = int(position_dst/7) - self.dxy[position_src][1]
            position_src = x + y * 7

            # 移動棋子，將棋子移到目的地的索引，然後將起始地歸零
            state.pieces[position_dst] = state.pieces[position_src]
            state.pieces[position_src] = 0

            # 若目的地上有敵方的棋子就取走變成我方的持駒
            piece_type = state.enemy_pieces[48-position_dst]
            if piece_type != 0:
                if piece_type != 4:
                    state.pieces[48+piece_type] += 1 # 持ち駒+1
                state.enemy_pieces[48-position_dst] = 0

        # 放回持駒 (將我方持有的持駒放回盤面上)
        else:
            capture = position_src-7
            state.pieces[position_dst] = capture
            state.pieces[48+capture] -= 1 # 持ち駒-1

        # 攻守交換，因為視角會轉換 (我方、敵方)，所以也要將棋子配置跟著轉換
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w
        return state

    # 判斷是否為先手
    def is_first_player(self):
        return self.depth%2 == 0

    # 顯示遊戲結果
    def __str__(self):
        # 根據先手、後手，會對調我方與敵方
        pieces0 = self.pieces  if self.is_first_player() else self.enemy_pieces
        pieces1 = self.enemy_pieces  if self.is_first_player() else self.pieces
        hzkr0 = ('', 'H', 'Z', 'K', 'R')
        hzkr1 = ('', 'h', 'z', 'k', 'r')

        # 後手的持駒區 (在盤面上面所以要先建立)
        str = '['
        for i in range(49, 52):
            if pieces1[i] >= 2: str += hzkr1[i-48]
            if pieces1[i] >= 1: str += hzkr1[i-48]
        str += ']\n'

        # 盤面
        for i in range(49):
            if pieces0[i] != 0:
                str += hzkr0[pieces0[i]]
            elif pieces1[48-i] != 0:
                str += hzkr1[pieces1[48-i]]
            else:
                str += '-'
            if i % 7 == 6:
                str += '\n'

        # 先手的持駒區 (在盤面下面所以要最後建立)
        str += '['
        for i in range(49, 52):
            if pieces0[i] >= 2: str += hzkr0[i-48]
            if pieces0[i] >= 1: str += hzkr0[i-48]
        str += ']\n'
        return str

# 隨機選擇動作
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# 單獨測試
if __name__ == '__main__':
    state = State()

    while True:
        if state.is_done():
            break

        state = state.next(random_action(state))

        print(state)
        print()