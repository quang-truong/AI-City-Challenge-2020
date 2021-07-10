# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import os.path as osp
import os

from vehiclereid.utils.iotools import check_isfile, mkdir_if_missing

def get_area(coord_list):
    sum = 0
    for box in coord_list:
        sum += box.area()
    return sum.item() if sum != 0 else 0

def crop_img(save_path, dir):
    for img_file in os.listdir(dir):
        img = cv2.imread(osp.join(dir, img_file))
        output = predictor(img)
        car_idx = []
        bus_idx = []
        truck_idx = []
        for i in range(output["instances"].pred_classes.size()[0]):
            if (output["instances"].pred_classes[i].item() == 2):
                car_idx.append(i)
            elif (output["instances"].pred_classes[i].item() == 5):
                bus_idx.append(i)
            elif (output["instances"].pred_classes[i].item() == 7):
                truck_idx.append(i)
        car_coord = [output["instances"].pred_boxes[i] for i in car_idx]
        bus_coord = [output["instances"].pred_boxes[i] for i in bus_idx]
        truck_coord = [output["instances"].pred_boxes[i] for i in truck_idx]
        if ((len(car_coord) == 0) and (len(bus_coord) == 0) and (len(truck_coord) == 0)):
            cv2.imwrite(osp.join(save_path, img_file), img)
            print("Cropped", osp.join(dir, img_file))
            continue
        tuple_car = (get_area(car_coord), car_coord)
        tuple_bus = (get_area(bus_coord), bus_coord)
        tuple_truck = (get_area(truck_coord), truck_coord)
        ls = [tuple_car, tuple_bus, tuple_truck]
        res = max(ls)
        max_area = res[1][0].area()
        final = res[1][0]
        for box in res[1]:
            if (max_area < box.area()):
                final = box
                max_area = box.area()
        minx1 = int(final.tensor[0, 0])
        maxx2 = int(final.tensor[0, 2])
        miny1 = int(final.tensor[0, 1])
        maxy2 = int(final.tensor[0, 3])
        if ((maxx2 - minx1)/(maxy2 - miny1) > 4 or (maxy2 - miny1)/(maxx2 - minx1) > 2.5):
            cv2.imwrite(osp.join(save_path, img_file), img)
            print("Cropped", osp.join(dir, img_file))
            continue
        cropped = img[miny1:maxy2, minx1:maxx2]
        cv2.imwrite(osp.join(save_path, img_file), cropped)
        print("Cropped", osp.join(dir, img_file))


if __name__ == "__main__":
    query_dir = "vehiclereid/datasets/AIC20_ReID/image_query"
    query_path = "vehiclereid/datasets/AIC20_ReID/image_query_cropped"
    train_dir = "vehiclereid/datasets/AIC20_ReID/image_train"
    train_path = "vehiclereid/datasets/AIC20_ReID/image_train_cropped"
    test_dir = "vehiclereid/datasets/AIC20_ReID/image_test"
    test_path = "vehiclereid/datasets/AIC20_ReID/image_test_cropped"
    mkdir_if_missing(query_path)
    mkdir_if_missing(train_path)
    mkdir_if_missing(test_path)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    crop_img(query_path, query_dir)
    crop_img(test_path, test_dir)
    crop_img(train_path, train_dir)
    print("Done cropping images!")
    