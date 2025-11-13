import torch.nn as nn

class Resnet34Backbone(nn.Module):
  def __init__(self, weights='DEFAULT'):
    super().__init__()

    from torchvision.models import resnet34
    
    resnet = resnet34(weights=weights)
    self.backbone = nn.Sequential(*list(resnet.children()[:-2]))