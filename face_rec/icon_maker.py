# -*- coding: utf-8 -*-
from PIL import Image, ImageFilter

# 打开图像
image = Image.open("./img/艾跃进_12.jpg")  # 替换为你的头像文件路径

# 像素化：缩小再放大
small_size = (16, 16)  # 调整这个值控制像素化程度
image_small = image.resize(small_size, resample=Image.NEAREST)
image_pixelated = image_small.resize(image.size, resample=Image.NEAREST)

# 应用模糊效果
blurry_image = image_pixelated.filter(ImageFilter.GaussianBlur(radius=5))  # 调整模糊半径

# 保存结果
blurry_image.save("pixelated_blurry_avatar.jpg")