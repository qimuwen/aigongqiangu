# -*- coding: utf-8 -*-
import os
from video_process.subtitle_orc import SubtitleExtractor
from face_rec.test_accuracy import DiscernFace


if __name__ == "__main__":
    orc_builder = SubtitleExtractor()
    dis_obj = DiscernFace()
    img_path = r"D:\13_pro\web\aigongqiangu\video_process\files"
    for i in os.listdir(img_path):
        file_path = f"{img_path}/{i}"
        text = orc_builder.process_image(file_path)
        result = dis_obj.discern(file_path)
        print(file_path, text, result)