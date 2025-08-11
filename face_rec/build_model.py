# -*- coding: utf-8 -*-
import face_recognition
import os
import numpy as np
from pathlib import Path
import pickle

# 训练数据文件夹路径
TRAINING_DIR = "./images/train"  # 替换为你的训练数据文件夹路径
MODEL_PATH = "./model/face_model.pkl"  # 模型保存路径
TEST_IMAGE = "./images/test/张维为_105.jpg"  # 替换为你的测试图片路径


def generate_model(training_dir, model_path):
    """生成并保存人脸识别模型"""
    known_face_encodings = []
    known_face_names = []

    # 遍历训练数据文件夹
    for image_path in Path(training_dir).glob("*.jpg"):
        # 从文件名提取人名（假设文件名格式为"人名_编号.jpg"）
        name = image_path.stem.split('_')[0]

        # 加载图片并获取人脸编码
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        # 确保图片中只有一张人脸
        if len(face_encodings) == 1:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)
        else:
            print(f"Warning: {image_path} contains {len(face_encodings)} faces, expected 1")

    if not known_face_encodings:
        print("No valid training data found. Model not generated.")
        return False

    # 保存模型
    model_data = {
        'encodings': known_face_encodings,
        'names': known_face_names
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_path}")
    return True


def main():
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print("No model found. Generating new model...")
        success = generate_model(TRAINING_DIR, MODEL_PATH)
        if not success:
            return


if __name__ == "__main__":
    main()
