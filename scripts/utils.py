import copy
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import List

def objects_threshold_scores(bboxes: torch.Tensor, 
                         labels: torch.Tensor=None, 
                         scores: torch.Tensor=None,
                         threshold_score: float=0.1):
    bboxes_copy = copy.deepcopy(bboxes)
    labels_copy = copy.deepcopy(labels)
    scores_copy = copy.deepcopy(scores)

    bboxes = torch.Tensor([])
    labels, scores = list(), list()
    for i, score in enumerate(scores_copy):
        if score >= threshold_score:
            bboxes = torch.cat((bboxes, bboxes_copy[i].unsqueeze(dim=0)), dim=0)
            labels.append(labels_copy[i])
            scores.append(score)
    
#     bboxes = torch.Tensor(bboxes).unsqueeze(dim=0)
    labels = torch.Tensor(labels)
    scores = torch.Tensor(scores)

    del bboxes_copy, labels_copy, scores_copy

    return bboxes, labels, scores

def show_image_with_objects(image: np.array, 
                            bboxes: torch.Tensor, 
                            colors: List,
                            labels: torch.Tensor=None, 
                            scores: torch.Tensor=None,
                            threshold_score: float=0.5):

    image = Image.fromarray(image.transpose(1, 2, 0))


#     random.shuffle(color)

    if scores != None:
        bboxes, labels, scores = objects_threshold_scores(bboxes, labels, scores, threshold_score)

    for i in range(len(bboxes)):
        draw = ImageDraw.Draw(image)
        draw.rectangle(bboxes[i].numpy(), outline = colors[labels[i].int()], width=2)

        if scores != None:
            bbox = draw.textbbox((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])} {scores[i] * 100:.2f}%")
            draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
            draw.text((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])} {scores[i] * 100:.2f}%", colors[labels[i].int()])
        else:
            bbox = draw.textbbox((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])}")
            draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
            draw.text((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])}", colors[labels[i]])
    return image