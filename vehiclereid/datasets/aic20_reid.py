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


class AIC20_ReID(BaseImageDataset):
    """
    AIC20_ReID

    Dataset statistics:
    # identities: 666 vehicles(333 for training and 333 for testing)
    # images: 36935 (train) + 18290 (test) + 1052 (query)
    """
    dataset_dir = 'AIC20_ReID'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super((AIC20_ReID), self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train_cropped')
        self.query_dir = osp.join(self.dataset_dir, 'image_query_cropped')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test_cropped')
        self.train_txt_path = osp.join(self.dataset_dir, "train_track_id.txt")
        self.train_xml_path = osp.join(self.dataset_dir, "train_label.xml")
        self.test_txt_path = osp.join(self.dataset_dir, "test_track_id.txt")

        self.check_before_run()

        train, gallery, query = self.process_dir()

        if verbose:
            print('=> AIC20_ReID loaded')
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
        if not osp.exists(self.train_txt_path):
            raise RuntimeError('"{}" is not available'.format(self.train_txt_path))
        if not osp.exists(self.train_xml_path):
            raise RuntimeError('"{}" is not available'.format(self.train_xml_path))
        if not osp.exists(self.test_txt_path):
            raise RuntimeError('"{}" is not available'.format(self.test_txt_path))

    def process_dir(self):
        xml_path = self.train_xml_path
        img_path = self.train_dir
        train_data = []
        query_data = []
        gallery_data = []
        doc = xml.dom.minidom.parse(xml_path)
        itemlist = doc.getElementsByTagName('Item')
        pid_container = set()
        for i in itemlist:
            pid_container.add(i.attributes['vehicleID'].value)
        train_container = set()
        test_container = set()
        pid_container = sorted(list(pid_container))
        for i in range(len(pid_container)):
            #print(pid_container[i], i, end= "\t")
            if (i in [0,1,2,6,8,9,11,162,221,313,329,21,284,30,99,92,60,170]):
                test_container.add(pid_container[i])
            else:
                train_container.add(pid_container[i])
        pid2label = {pid: label for label, pid in enumerate(train_container)}
        query_container = set()
        #track_ls = [False for i in range(1897)] # Length of track list
        #with open("train_pid2track.pkl", "rb") as f:
        #    train_pid2track = pickle.load(f)
        for i in itemlist:
            if (i.attributes['vehicleID'].value in train_container):
                #if (track_ls[train_pid2track[]]
                train_data.append((
                    osp.join(img_path, i.attributes['imageName'].value), 
                    pid2label[i.attributes['vehicleID'].value], 
                    int(i.attributes['cameraID'].value[1:])))
            else:
                if (i.attributes['vehicleID'].value not in query_container):
                    query_container.add(i.attributes['vehicleID'].value)
                    query_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        int(i.attributes['vehicleID'].value), 
                        int(i.attributes['cameraID'].value[1:])))
                else:
                    gallery_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        int(i.attributes['vehicleID'].value), 
                        int(i.attributes['cameraID'].value[1:])))

        return train_data, gallery_data, query_data

