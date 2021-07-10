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


class AIC20_ReID_CamID(BaseImageDataset):
    """
    AIC20_ReID

    Dataset statistics:
    # identities: 666 vehicles(333 for training and 333 for testing)
    # images: 36935 (train) + 18290 (test) + 1052 (query)
    """
    dataset_dir = 'AIC20_ReID'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super((AIC20_ReID_CamID), self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_txt_path = osp.join(self.dataset_dir, "train_track_id.txt")
        self.train_xml_path = osp.join(self.dataset_dir, "train_label.xml")
        self.test_txt_path = osp.join(self.dataset_dir, "test_track_id.txt")

        self.check_before_run()

        train, gallery, query = self.process_dir()

        if verbose:
            print('=> AIC20_ReID_CamID loaded')
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
        
        cid2label = {'c019':21, 'c003': 33, 'c023': 30, 'c016': 0, 'c011': 10, 'c024': 31, 'c038': 22, 'c012': 6, 'c020': 34, 'c015': 17, 'c018': 12, 'c026': 35, 'c002': 3, 'c021': 18, 'c031': 32, 'c022': 9, 'c005': 29, 'c040': 19, 'c036': 14, 'c010': 23, 'c028': 26, 'c014': 2, 'c027': 8, 'c037': 20, 'c025': 1, 'c001': 11, 'c017': 25, 'c035': 27, 'c033': 5, 'c029': 4, 'c034': 28, 'c030': 24, 'c004': 7, 'c039': 13, 'c013': 16, 'c032': 15}

        test_dict = {}
        query_dict = {}
        for i in itemlist:
            if (cid2label[i.attributes['cameraID'].value] not in query_dict):
                query_dict[cid2label[i.attributes['cameraID'].value]] = 1
                query_data.append((
                    osp.join(img_path, i.attributes['imageName'].value), 
                    cid2label[i.attributes['cameraID'].value], 
                    int(i.attributes['cameraID'].value[1:])))
            else:
                if (cid2label[i.attributes['cameraID'].value] not in test_dict):
                    test_dict[cid2label[i.attributes['cameraID'].value]] = 1
                    gallery_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        cid2label[i.attributes['cameraID'].value], 
                        int(i.attributes['cameraID'].value[1:])))
                elif (test_dict[cid2label[i.attributes['cameraID'].value]] <= 4):
                    test_dict[cid2label[i.attributes['cameraID'].value]] += 1
                    gallery_data.append(
                        (osp.join(img_path, i.attributes['imageName'].value), 
                        cid2label[i.attributes['cameraID'].value], 
                        int(i.attributes['cameraID'].value[1:])))
                else:
                    train_data.append((
                        osp.join(img_path, i.attributes['imageName'].value), 
                        cid2label[i.attributes['cameraID'].value], 
                        int(i.attributes['cameraID'].value[1:])))

        return train_data, gallery_data, query_data

