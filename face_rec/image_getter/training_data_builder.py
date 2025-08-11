# -*- coding: utf-8 -*-
import os
import random
import shutil
from collections import defaultdict

BASE_PATH = "../"
TRAIN_PATH = os.path.join(BASE_PATH, "images", "train")  # 修正拼写：tarin -> train
TEST_PATH = os.path.join(BASE_PATH, "images", "test")

IMG_PATH = "../img"


def init_path():
    # 确保父目录 images 存在
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)

    # 创建 train 和 test 目录
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)
    print(f"Created directories: {TRAIN_PATH}, {TEST_PATH}")


def copy_file():
    test_file_count = 10  # 测试数据集数量
    min_train_count = 100  # 最少数据集设置
    file_list = [i for i in os.listdir(IMG_PATH)]
    image_set = set()
    image_count = {}
    for file in os.listdir(IMG_PATH):
        name = file.split("_")[0]
        image_set.add(name)
        if name not in image_count:
            image_count[name] = 0
        image_count[name] += 1
    for name, count in image_count.items():
        if count < min_train_count:
            ret = f"{name} image 数量太少(<{min_train_count}),目前数量{count}"
            print(ret)
            raise
    name_to_files = defaultdict(list)
    for file in file_list:
        name = file.split('_')[0]
        name_to_files[name].append(file)
    for name, files in name_to_files.items():
        print(f"Processing {name}: {len(files)} images")
        # 随机选择10张（如果少于10张，全部选到测试集）
        test_files = random.sample(files, min(test_file_count, len(files)))
        train_files = [f for f in files if f not in test_files]

        # 复制到测试目录
        for file in test_files:
            src = os.path.join(IMG_PATH, file)
            dst = os.path.join(TEST_PATH, file)
            shutil.copy(src, dst)
            print(f"Copied to test: {dst}")

        # 复制到训练目录
        for file in train_files:
            src = os.path.join(IMG_PATH, file)
            dst = os.path.join(TRAIN_PATH, file)
            shutil.copy(src, dst)
            print(f"Copied to train: {dst}")


if __name__ == "__main__":
    init_path()
    copy_file()
