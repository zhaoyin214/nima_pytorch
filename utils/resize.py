#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   resize.py
@time    :   2019/09/10 16:56:28
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""


#%%
from skimage.transform import rescale


#%%
def resize_image(image, large_side=640):

    image_height, image_width = image.shape[: 2]
    if image_width < image_height:
        scale = large_side / image_height
    else:
        scale = large_side / image_width

    image = rescale(image, scale=scale, multichannel=True)

    return image


#%%
if __name__ == "__main__":

    from skimage.io import imread, imsave
    import warnings
    # 310261, 442939
    image_id = "852750"
    image = imread("./" + image_id + "_.jpg")
    image = resize_image(image)
    # imsave("./" + image_id + ".jpg", image)
    try:
        warings.warn(imsave("./" + image_id + ".jpg", image))
    except:
        print("warning")

#%%
