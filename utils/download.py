#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   download.py
@time    :   2019/09/10 16:47:24
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""


#%%
import bs4
import urllib
import os
import time


#%%
def download_image(image_id, image_root=None):

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "compress",
        "Accept-Language": "en-us;q=0.5,en;q=0.3",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0"
        }

    url = "https://www.dpchallenge.com/image.php?IMAGE_ID={}".format(image_id)
    req = urllib.request.Request(url=url, headers=headers)
    response = urllib.request.urlopen(req)
    html = response.read()
    # html = html.decode("utf-8")
    soup = bs4.BeautifulSoup(html, "lxml")
    time.sleep(3)

    for item in soup.find_all("td", attrs={"id": "img_container"}):

        image_url = "https:" + item.find_all("img")[1]["src"]

        if image_root is None:
            print(image_url)
            continue

        with open(os.path.join(image_root, str(image_id) + "_.jpg"), "wb") as f:
            # req = urllib.request.Request(url=image_url, headers=headers)
            f.write((urllib.request.urlopen(image_url)).read())

    time.sleep(3)

    return None

#%%
if __name__ == "__main__":

    # 310261, 442939, 689717, 776294, 878854, 639996, 759187, 932676, 677234, 2144, 415358,
    # 913289, 648325, 502422, 946863, 881798, 532402,
    # download_image(image_id=310261, image_root="./append")

    import pandas as pd
    from configs import AVA_ABNORMAL

    abnormals = pd.read_csv(
        filepath_or_buffer=AVA_ABNORMAL, header=0, index_col=0
    )

    for _, item in abnormals.iterrows():

        print("downloading: {}".format(item.image_id))
        try:
            download_image(image_id=item.image_id, image_root="./append")
        except:
            continue

#%%
