# -*- coding: utf-8 -*-

import requests
import re
import time
from urllib import parse
import random
import os

IMG_PATH = "../img"


def init_path():
    os.mkdir(IMG_PATH)


def main(target, target_txt):
    page = 1
    count = 1
    for w in range(0, 5):
        url = f"https://pic.sogou.com/d?query={target}&forbidqc=&entityid=&preQuery=&rawQuery=&queryList=&st=&did={page}"
        page += 60
        url_list = get_html(url)
        reurl = save_url(url_list)
        for i in reurl:
            pt = requests.get(i)
            time.sleep(random.random() * 3)
            with open(f"{IMG_PATH}/{target_txt}_{count}" + ".jpg", "wb+") as file:
                file.write(pt.content)
                file.close()
                print("抓取完成：", count)
                count += 1


def get_html(url):  # 一次请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    return response.text


def save_url(baseurl):  # 获取真实的html
    find_link = re.compile(r'&amp;url=(.*?)" alt="')
    cid = re.findall(find_link, baseurl)
    return cid


if __name__ == '__main__':
    txt = "张维为"
    url_txt = parse.quote(txt)
    main(url_txt, txt)
