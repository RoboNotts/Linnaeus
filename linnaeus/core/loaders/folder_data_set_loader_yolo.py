from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
import os
from linnaeus.core.data_augmentation import preprocessing
from pathlib import Path

class FolderDataSetLoaderYolo(Dataset):
    def __init__(self, path, classes):
        """
        self-defined dataset data in yolo format
        """
        super(FolderDataSetLoaderYolo, self).__init__()
        self.dataset_path = Path(path)
        # root of images
        self.root_images = self.dataset_path / "images"
        # root of annotations
        self.root_labels = self.dataset_path / "labels"
        # load names of labels
        self.labels = os.listdir(self.root_labels)
        # list of classes
        self.classes = classes

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        # obtain the name of the label
        label_name = self.labels[index]
        # load annotations
        fh = open(self.root_labels / label_name, 'r')
        tags = []
        for line in fh:
            line = line.strip('\n')
            label = line.split(' ')
            x, y, w, h = 480 * float(label[1]), 360 * float(label[2]), 480 * float(label[3]), 360 * float(label[4])
            xmin = x - 0.5 * w
            ymin = y - 0.5 * h
            xmax = x + 0.5 * w
            ymax = y + 0.5 * h
            tags.append(torch.Tensor([int(label[0]), xmin, ymin, xmax, ymax]))
        fh.close()
        # obtain the name of the image
        image_name = label_name.split(".")[0] + ".jpg"
        # load image
        image = cv2.imread(str(self.root_images / image_name))
        # image preprocessing
        torch_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        torch_image = preprocessing(torch_image)
        return torch_image, tags

    @staticmethod
    def collate_fn(batch):
        images = list()
        tag_batches = list()

        for (image, tags) in batch:
            images.append(image)
            tag_batches.append(tags)
        
        return torch.stack(images, dim=0), tag_batches

    def __len__(self):
        return len(self.labels)
