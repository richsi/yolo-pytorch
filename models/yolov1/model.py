import torch
import torch.nn as nn
from models.common.backbones import Resnet34Backbone

class YOLOv1(nn.Module):
  def __init__(self, num_boxes=2, num_classes=20):
    super().__init__()

    # everything except last two layers
    self.backbone = Resnet34Backbone()
    self.backbone.requires_grad_(False) # freeze backbone weights

    device = "gpu" if torch.cuda.is_available() else "cpu"

    self.detection_head = nn.Sequential(
      # 14x14x512 -> 7x7x512
      nn.AdaptiveAvgPool2d((7,7)),
      # 7x7x512 -> 25088
      nn.Flatten(),
      # 25088 -> 4096
      nn.Linear(in_features=(7 * 7 * 512), out_features=4096, device=device),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Dropout(p=0.5),
      # 4096 -> 
      nn.Linear(in_features=4096, out_features=(7 * 7 * (5 * num_boxes + num_classes))),
      nn.Sigmoid()
    )

    self.num_boxes = num_boxes
    self.num_classes = num_classes

  def forward(self, x):
    features = self.backbone(x)
    output = self.detection_head(features)

    # reshape to YOLO format: (batch, 7, 6, 30)
    # output = output.view(-1, 7, 7, 5 * self.num_boxes + self.num_classes)

    return output