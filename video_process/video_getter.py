# -*- coding: utf-8 -*-
from you_get import common
url = "https://www.bilibili.com/video/BV1Yf421Q7JJ/?spm_id_from=333.337.search-card.all.click&vd_source=aa47096193f82ac1eaba3b9f263dbded"
common.any_download(url, cookies="cookies.txt", output_dir="./videos", merge=True)

