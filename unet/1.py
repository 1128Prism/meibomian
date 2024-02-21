import numpy as np
from PIL import Image

# 假设prediction是模型预测的结果，值为0、1、2等类别标签
# 使用你的模型得到的预测结果替换下面的示例数据
prediction = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

# 创建一个颜色映射数组
palette = [
    0, 0, 0,      # 类别0对应黑色
    0, 255, 0,    # 类别1对应绿色
    0, 0, 255     # 类别2对应蓝色
]

# 将预测结果映射到对应的颜色值
colored_mask = Image.fromarray(prediction.astype(np.uint8))
colored_mask.putpalette(palette)  # 将颜色映射应用到图像

# 显示图像
colored_mask.show()

