from xml.dom.minidom import parse as xml_parse
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from fcos.core.data_augmentation import preprocessing

class FolderDataSetLoader(Dataset):

    def __init__(self, path, classes):
        super().__init__()
        
        self.dataset_path = Path(path)
        self.classes = classes

        # initialize a list to save input images as torch tensors
        self._dataset = list(self.dataset_path.glob("*.xml"))

    def __getitem__(self, index):
        # Return label of an image corresponding to its index.# read annotation file
        label = self._dataset[index]

        dom = xml_parse(str(label.resolve()))
        # obtain root of the xml file
        root = dom.documentElement

        # Get image from provided path
        path = root.getElementsByTagName('path')[0].childNodes[0].data
        image = cv2.imread(str((self.dataset_path / path).resolve()))
    
        # obtain image size
        row = image.shape[0]
        col = image.shape[1]
        # image preprocessing
        torch_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        torch_image = preprocessing(torch_image)
        objects = root.getElementsByTagName("object")
        # analyse annotations
        tags = {c:[] for c in self.classes}
        for obj in objects:
            bndbox = obj.getElementsByTagName('bndbox')[0]
            name = obj.getElementsByTagName('name')[0]
            name_data = name.childNodes[0].data
            
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = int(float(xmin.childNodes[0].data))
            
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = int(float(ymin.childNodes[0].data))
            
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = int(float(xmax.childNodes[0].data))
            
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = int(float(ymax.childNodes[0].data))

            # obtain top left anf right bottom coordinate
            left = int(480 * xmin_data / col)
            top = int(360 * ymin_data / row)
            right = int(480 * xmax_data / col)
            bottom = int(360 * ymax_data / row)
            l = [left, top, right, bottom]
            tags[name_data].append(l)
        tag_list = [torch.Tensor([self.classes.index(k), *vv]) for k, v in tags.items() for vv in v if len(vv) > 0]
        return torch_image, tag_list

    def __len__(self):
        return len(self._dataset)

    @staticmethod
    def collate_fn(batch):
        images = list()
        tag_batches = list()

        for (image, tags) in batch:
            images.append(image)
            tag_batches.append(tags)
        
        return torch.stack(images, dim=0), tag_batches
