# -*- coding: utf-8 -*-
import os
import re
FILE_PATH = "./videos"

find_link = re.compile(r"P(\d*).")
def judge_rename(file):
    if file.startswith("艾跃进——为"):
        page = re.findall(find_link, file)[0]
        os.rename(os.path.join(FILE_PATH, file), os.path.join(FILE_PATH, f"为人民服务是毛泽东思想的精华_{page}.mp4"))


if __name__ == "__main__":
    for _file in os.listdir(FILE_PATH):
        judge_rename(_file)