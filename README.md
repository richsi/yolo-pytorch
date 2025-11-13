## yolo-pytorch

### YOLOv1
- Use ResNet34 ImageNet weights to avoid pretraining
  - ResNet34 outputs 512-channel feature maps at 7x7 (for 224x224 inputs), which matches YOLOv1's requirements nicely
  - Remove global average pooling and classification layers
- Adjust YOLOv1 attention head accordingly since backbone will output 14x14 if using 448x448 input
