import torch
import torch.nn as nn


class YOLOv1Loss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.S = config.model.detection_head.grid_size
        self.B = config.model.detection_head.num_boxes
        self.C = config.data.num_classes
        self.lambda_coord = config.loss.lambda_coord
        self.lambda_noobj = config.loss.lambda_noobj

    def compute_iou(self, boxes1, boxes2):
        """
        Computes IoU between two sets of boxes.

        IoU = area over overlap / area of union

        boxes1: (N, 4) [x_center, y_center, width, height]
        boxes2: (M, 4) [x_center, y_center, width, height]
        Returns: (N, M) IoU Matrix
        """
        # x1, y1 is top left
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

        # get intersection coordinates
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height

        # union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # (N) -> (N, 1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # (M) -> (1, M)
        union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area

        return inter_area / (union_area + 1e-6)

    def forward(self, predictions, targets):
        """
        predictions: (batch, S*S*(B*5+C)) - raw model output
        targets: (batch, S*S*(B*5+C)) - ground truth labels
        """

        batch_size = predictions.size(0)

        # reshape predictions to (batch, S, S, B*5+C)
        predictions = predictions.view(batch_size, self.S, self.S, self.B * 5 + self.C)

        # split predictions 
        # boxes: (batch, S, S, B, 5)
        pred_boxes = predictions[..., : self.B * 5].view(
            batch_size, self.S, self.S, self.B, 5
        )
        pred_x = torch.sigmoid(pred_boxes[..., 0])  # center x (0-1)
        pred_y = torch.sigmoid(pred_boxes[..., 1])  # center x (0-1)
        pred_w = pred_boxes[..., 2]  # raw width
        pred_h = pred_boxes[..., 3]  # raw height
        pred_conf = pred_boxes[..., 4]  # confidence (0-1)
        pred_class = predictions[..., self.B * 5 :]

        # split targets
        target_boxes = predictions[..., : self.B * 5].view(
            self.S, self.S, self.B * 5 + self.C
        )
        target_x = target_boxes[..., 0]
        target_y = target_boxes[..., 1]
        target_w = target_boxes[..., 2]
        target_h = target_boxes[..., 3]
        target_conf = target_boxes[..., 4] # 1.0 for object cells, 0 for empty
        target_class = target_boxes[..., self.B * 5:]

        # create object mask (batch, S, S, 1)
        obj_mask = (target_conf.sum(dim=-1, keepdim=True) > 0).float()