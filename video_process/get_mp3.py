# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# video2mp3.py
import sys
import os
import ffmpeg

def extract_audio_to_mp3(video_path, mp3_path=None, bitrate="192k"):
    """把视频文件提取为 mp3"""
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    # 如果没给输出路径，默认保存在视频同目录同名.mp3
    if mp3_path is None:
        root, _ = os.path.splitext(video_path)
        mp3_path = root + ".mp3"

    # 构建 ffmpeg 命令：-vn 去掉视频流，-acodec libmp3lame 用 mp3 编码器
    stream = (
        ffmpeg
        .input(video_path)
        .output(mp3_path, acodec="libmp3lame", audio_bitrate=bitrate, vn=None)
        .overwrite_output()
    )

    print(f"Extracting audio → {mp3_path}")
    ffmpeg.run(stream, quiet=True)
    print("Done!")

if __name__ == "__main__":
    input_path = "./videos"
    output_path = "./audio"
    for file in os.listdir(input_path):

        extract_audio_to_mp3(os.path.join(input_path, file), os.path.join(output_path, file.split(".")[0] + ".mp3"))