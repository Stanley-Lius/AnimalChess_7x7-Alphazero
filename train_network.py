# ====================
# 訓練模組
# ====================

# 匯入套件
from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

# 設定參數
RN_EPOCHS = 100  # 訓練次數

# 載入訓練資料
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# 訓練對偶網路
def train_network():
    # Step01 載入訓練資料
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # Step02 重塑訓練資料的 shape 以進行訓練
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # Step03 載入最佳玩家的模型
    model = load_model('./model/best.h5')

    # Step04 編譯模型
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # Step05 設定要調整的學習率
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # Step06 利用 callbacks 印出訓練的次數
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
                print('\rTrain {}/{}'.format(epoch + 1,RN_EPOCHS), end=''))

    # Step07 進行訓練
    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=RN_EPOCHS,
            verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    # Step08 儲存最新玩家的模型
    model.save('./model/latest.h5')

    # Step09 清除 session 與刪除模型
    K.clear_session()
    del model

# 單獨測試
if __name__ == '__main__':
    train_network()

