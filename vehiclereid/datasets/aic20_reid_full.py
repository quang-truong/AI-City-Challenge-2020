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


class AIC20_ReID_Full(BaseImageDataset):
    """
    AIC20_ReID

    Dataset statistics:
    # identities: 666 vehicles(333 for training and 333 for testing)
    # images: 36935 (train) + 18290 (test) + 1052 (query)
    """
    dataset_dir = 'AIC20_ReID'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super((AIC20_ReID_Full), self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train_cropped')
        self.query_dir = osp.join(self.dataset_dir, 'image_query_cropped')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test_cropped')
        self.train_xml_path = osp.join(self.dataset_dir, "train_label.xml")
        self.gallery_txt_path = osp.join(self.dataset_dir, "name_test.txt")
        self.query_txt_path = osp.join(self.dataset_dir, "name_query.txt")

        self.check_before_run()

        train = self.process_train_dir()
        query = self.process_query_dir(self.query_txt_path, self.query_dir)
        #gallery = self.process_test_dir(self.dataset_dir, self.gallery_dir)
        gallery = self.process_query_dir(self.gallery_txt_path, self.gallery_dir)

        if verbose:
            print('=> AIC20_ReID_Full loaded')
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
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))
        if not osp.exists(self.gallery_txt_path):
            raise RuntimeError('"{}" is not available'.format(self.gallery_txt_path))
        if not osp.exists(self.train_xml_path):
            raise RuntimeError('"{}" is not available'.format(self.train_xml_path))
        if not osp.exists(self.query_txt_path):
            raise RuntimeError('"{}" is not available'.format(self.query_txt_path))

    def process_train_dir(self):
        xml_path = self.train_xml_path
        img_path = self.train_dir
        train_data = []
        doc = xml.dom.minidom.parse(xml_path)
        itemlist = doc.getElementsByTagName('Item')
        pid_container = set()
        for i in itemlist:
            pid_container.add(i.attributes['vehicleID'].value)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for i in itemlist:
            train_data.append((
                osp.join(img_path, i.attributes['imageName'].value), 
                pid2label[i.attributes['vehicleID'].value], 
                int(i.attributes['cameraID'].value[1:])))
        return train_data


    def process_query_dir(self, text, dir):
        f = open(text, 'r')
        lines = f.readlines()
        dataset = []
        for i in lines:
            dataset.append((
                osp.join(dir, i)[:-1],
                int(i[:6]),
                0,
            ))
        f.close()
        return dataset

    def process_test_dir(self, dataset_dir, dir):
        dataset = []
        with open(osp.join(dataset_dir, "test_images.pkl"), 'rb') as f:
            test_images = pickle.load(f)
        for i in test_images:
            dataset.append((
                osp.join(dir, i),
                int(i[:6]),
                0,
            ))
        return dataset