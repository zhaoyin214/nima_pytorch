import warnings
from PIL import ImageFile
from skimage.io import imread
import os
from download import download_image

# warnings.filterwarnings("error")
# ImageFile.LOAD_TRUNCATED_IMAGES = True


def _download(image_id, image_root):

    print("image {} is downloading...".format(image_id))
    image_root = os.path.join(image_root, "append")
    download_image(image_id=image_id, image_root=image_root)
    image_filepath = os.path.join(image_root, image_id)
    image = imread(fname=image_filepath + "_.jpg")
    image = resize_image(
        image=image, large_side=self._image_large_side
    )
    imsave(fname=image_filepath + ".jpg", arr=image)
    # image_filepath = os.path.join(self._image_root, image_id + ".jpg")
    # imsave(fname=image_filepath, arr=image)

if __name__ == "__main__":
    image_id = "310261"
    # image_id = "532402"
    # image_id = "53"
    image_root = "e:/src/jupyter/datasets/AVA/images"
    image_filepath = os.path.join(image_root, image_id + ".jpg")
    print(imread(fname=image_filepath).shape)
    # print("image id: {}".format(image_id))

    # if os.path.isfile(image_filepath):
    #     try:
    #         warnings.warn(imread(fname=image_filepath))
    #     except:
    #         _download(image_id=image_id, image_root=image_root)
    # else:
    #     _download(image_id=image_id, image_root=image_root)
