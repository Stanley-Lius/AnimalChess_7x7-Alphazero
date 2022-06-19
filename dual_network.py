# ====================
#建立對偶網路
# ====================

# 匯入套件
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# 設定參數
DN_FILTERS  = 128 # 卷積層的卷積核數
DN_RESIDUAL_NUM =  16 # 殘差塊數
DN_INPUT_SHAPE = (7, 7, 14) # 輸入 shape，3x4 為盤面大小，14 為我方與敵方的配置
DN_OUTPUT_SIZE = 539 # 動作數（動作公式算出來的總動作數）

# 畳み込み層の作成
def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

# 建構卷積層
def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f

# 建立對偶網路
def dual_network():
    if os.path.exists('./model/best.h5'):
        return

    # 輸入層
    input = Input(shape=DN_INPUT_SHAPE)

    # 卷積層
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 殘差塊×16
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # 池化層
    x = GlobalAveragePooling2D()(x)

    # 策略輸出
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
                activation='softmax', name='p')(x)

    # 價值輸出
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # 建構模型
    model = Model(inputs=input, outputs=[p,v])

    # 儲存模型
    os.makedirs('./model/', exist_ok=True)
    model.save('./model/best.h5')

    # 刪除模型
    K.clear_session()
    del model

# 單獨測試
if __name__ == '__main__':
    dual_network()
