# -*- coding: utf-8 -*-
import os

import whisper

# 加载模型（可选择 tiny, base, small, medium, large）
model = whisper.load_model("medium")  # 推荐 medium，精度高且资源需求适中


# 保存为 SRT 文件
def save_srt(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def format_timestamp(seconds):
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


if __name__ == "__main__":
    audio_path = "./audio"
    srt_path = "./srt"
    for file in os.listdir(audio_path):
        file_path = os.path.join(audio_path, file)
        # 转录音频文件
        result = model.transcribe(
            file_path,  # 支持 mp3, wav, mp4 等格式
            language="zh",  # 指定语言（可选，自动检测效果也不错）
            task="transcribe",  # 或 "translate"（翻译到英语）
            verbose=True,  # 显示转录进度
            word_timestamps=True  # 启用逐词时间戳
        )
        srt_file = os.path.join(srt_path, file[:-3] + "srt")
        # 保存字幕到文件
        save_srt(result["segments"], srt_file)
