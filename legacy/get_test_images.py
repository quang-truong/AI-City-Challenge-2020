import os.path as osp
import os
import pickle

from vehiclereid.utils.iotools import check_isfile, mkdir_if_missing
import shutil

test_dir = "vehiclereid/datasets/AIC20_ReID/image_test_cropped/"
mkdir_if_missing("vehiclereid/datasets/AIC20_ReID/image_test_repr/")
with open("test_images.pkl", "rb") as f:
    test_imgs = pickle.load(f)

for img in test_imgs:
    src = test_dir + img
    dst = "vehiclereid/datasets/AIC20_ReID/image_test_repr/" + img
    shutil.copy(src, dst)
