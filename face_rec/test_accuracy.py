# -*- coding: utf-8 -*-
import face_recognition
import os
import numpy as np
from pathlib import Path
import pickle

TEST_PATH = "./images/test"
MODEL_PATH = "./model/face_model.pkl"  # 模型保存路径
TEST_IMAGE = "./images/test/张维为_105.jpg"  # 替换为你的测试图片路径


def load_model(model_path):
    """加载已保存的人脸识别模型"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['encodings'], model_data['names']
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return None, None


def recognize_face(test_image_path, known_face_encodings, known_face_names):
    """使用模型识别测试图片中的人脸"""
    if not known_face_encodings or not known_face_names:
        return "Model is empty or not loaded"

    # 加载测试图片
    test_image = face_recognition.load_image_file(test_image_path)
    test_face_encodings = face_recognition.face_encodings(test_image)

    if len(test_face_encodings) == 0:
        return "No faces found in the test image"

    if len(test_face_encodings) > 1:
        return "Multiple faces found in the test image"

    # 获取测试图片的人脸编码
    test_encoding = test_face_encodings[0]

    # 比较人脸并找到最佳匹配
    distances = face_recognition.face_distance(known_face_encodings, test_encoding)
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    # 设置阈值以确保匹配可靠性
    if min_distance < 0.6:  # face_recognition库建议的阈值
        return known_face_names[min_distance_index], min_distance
    else:
        return "Unknown person", min_distance


def test():
    print("Loading model...")
    known_face_encodings, known_face_names = load_model(MODEL_PATH)

    if not known_face_encodings:
        print("Failed to load model. Exiting.")
        return

    print(f"Loaded model with {len(known_face_encodings)} known faces.")
    for file in os.listdir(TEST_PATH):
        # 识别测试图片中的人脸
        result = recognize_face(f"{TEST_PATH}/{file}", known_face_encodings, known_face_names)
        print(f"Recognition result: {file, result}")


if __name__ == "__main__":
    test()
