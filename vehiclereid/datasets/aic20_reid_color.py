from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import os
import xml.dom.minidom
import pickle

from .base import BaseImageDataset


class AIC20_ReID_Color(BaseImageDataset):
    """
    AIC20_ReID

    Dataset statistics:
    # identities: 666 vehicles(333 for training and 333 for testing)
    # images: 36935 (train) + 18290 (test) + 1052 (query)
    """
    dataset_dir = 'AIC20_ReID'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super((AIC20_ReID_Color), self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train_cropped')
        self.train_txt_path = osp.join(self.dataset_dir, "train_track_id.txt")
        self.train_xml_path = osp.join(self.dataset_dir, "train_label.xml")
        with open("train_pid2color.pickle", "rb") as f:
            self.pid2color = pickle.load(f)

        self.check_before_run()

        train, gallery, query = self.process_dir()

        if verbose:
            print('=> AIC20_ReID_Color loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.train_txt_path):
            raise RuntimeError('"{}" is not available'.format(self.train_txt_path))
        if not osp.exists(self.train_xml_path):
            raise RuntimeError('"{}" is not available'.format(self.train_xml_path))

    def process_dir(self):
        color_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
        xml_path = self.train_xml_path
        img_path = self.train_dir
        train_data = []
        query_data = []
        gallery_data = []
        doc = xml.dom.minidom.parse(xml_path)
        itemlist = doc.getElementsByTagName('Item')
        test_dict = {}
        query_dict = {}
        for i in itemlist:
            if (color_dict[self.pid2color[i.attributes['imageName'].value]] not in query_dict):
                query_dict[color_dict[self.pid2color[i.attributes['imageName'].value]]] = 1
                query_data.append((
                    osp.join(img_path, i.attributes['imageName'].value), 
                    color_dict[self.pid2color[i.attributes['imageName'].value]], 
                    int(i.attributes['cameraID'].value[1:])))
            else:
                if (color_dict[self.pid2color[i.attributes['imageName'].value]] not in test_dict):
                    test_dict[color_dict[self.pid2color[i.attributes['imageName'].value]]] = 1
                    gallery_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        color_dict[self.pid2color[i.attributes['imageName'].value]], 
                        int(i.attributes['cameraID'].value[1:])))
                elif (test_dict[color_dict[self.pid2color[i.attributes['imageName'].value]]] <= 30):
                    test_dict[color_dict[self.pid2color[i.attributes['imageName'].value]]] += 1
                    gallery_data.append(
                        (osp.join(img_path, i.attributes['imageName'].value), 
                        color_dict[self.pid2color[i.attributes['imageName'].value]], 
                        int(i.attributes['cameraID'].value[1:])))
                else:
                    train_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        color_dict[self.pid2color[i.attributes['imageName'].value]], 
                        int(i.attributes['cameraID'].value[1:])))

        return train_data, gallery_data, query_data


