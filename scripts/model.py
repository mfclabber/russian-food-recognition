import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from typing import Tuple, Dict, List

class FasterRCNN_ResNet50(torch.nn.Module):
  def __init__(self, num_classes: int=127) -> None:
    super().__init__()

    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    num_classes = num_classes + 1
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    for child in list(self.model.children())[:-1]:
      for param in child.parameters():
          param.requires_grad = False

  def predict(self, X: torch.Tensor) -> torch.Tensor:
    '''
    For predict bboxes and labels
    '''
    return self.model(X)

  # To calculate the loss function
  def forward(self, images: List[torch.Tensor], annotation: List[Dict[str, torch.Tensor]]) -> Dict[str, int]:
    return self.model(images, annotation)
  
class SSD_MobileNetV3(torch.nn.Module):
    def __init__(self, num_classes: int=127) -> None:
        super().__init__()

        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, 
                                                                                pretrained_backbone=False, 
                                                                                num_classes=num_classes + 1)
        self.num_classes = num_classes + 1

        for child in list(self.model.children()):
              for param in child.parameters():
                    param.requires_grad = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        For predict bboxes and labels
        '''
        return self.model(X)

    # To calculate the loss function
    def forward(self, images: List[torch.Tensor], annotation: List[Dict[str, torch.Tensor]]) -> Dict[str, int]:
        return self.model(images, annotation)