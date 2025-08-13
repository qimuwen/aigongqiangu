# -*- coding: utf-8 -*-
from you_get import common
for i in range(1, 57):
    url = f"https://www.bilibili.com/video/BV12UtmzAEFU/?spm_id_from=333.337.search-card.all.click&vd_source=aa47096193f82ac1eaba3b9f263dbded&p={i}"
    common.any_download(url, cookies="cookies.txt", output_dir="./videos", merge=True)

