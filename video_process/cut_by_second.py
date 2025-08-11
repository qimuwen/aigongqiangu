# -*- coding: utf-8 -*-
import cv2
import os

# 输入视频路径
video_path = "./videos/nobaby.mp4"
output_dir = "./files"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
frame_interval = int(fps)  # 每秒取一帧

count = 0
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 每隔 frame_interval 帧保存一张图片
    if count % frame_interval == 0:
        output_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_number += 1
    count += 1

# 释放资源
cap.release()
print(f"提取完成，共保存 {frame_number} 张图片")