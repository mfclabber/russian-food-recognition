import copy
import numpy as np
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader




class AlfaFoodDataset(Dataset):
    def __init__(self, images: List, objects: List[Dict[str, List]], transform: torchvision.transforms=None) -> None:
        self.images = images
        self.annotations = copy.deepcopy(objects)
        self.transform = transform
        self.num_classes = len(set(i for ob in objects for i in ob['categories']))
        self.list_transforms = np.zeros(shape=(len(self.images),))

        for i in range(len(self.annotations)):
            self.bboxes = self.annotations[i]['bbox']
            for bbox in self.bboxes:
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[Tuple[int]], Tuple[int]]:
        "Returns one sample of data: image, labels, bboxes"

        image = np.array(self.images[index].convert('RGB'))
        bboxes = self.annotations[index]['bbox']
        labels = self.annotations[index]['categories']

        if self.transform:
            # print(image.shape)
            transformed = self.transform(image = image, bboxes = bboxes, labels = labels)
            image = np.array(transformed['image']).transpose(1, 2, 0)
            bboxes = transformed['bboxes']
            labels = transformed['labels']

            self.list_transforms[index] = 1
        image = image.transpose(2, 0, 1)
        target = dict()
        target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        if target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.Tensor([0, 0, 1e-10, 1e-10]).unsqueeze(dim=0)
        if target['labels'].shape == torch.Size([0]):
            target['labels']= torch.zeros(size=(1, ), dtype=torch.int64)
        return image, target


    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.images)