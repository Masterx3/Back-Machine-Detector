import torch
import torch.nn as nn
from GymDetector.utils.main_utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits for objectness score predictions
        self.mse = nn.MSELoss()  # Mean Squared Error loss for box coordinate predictions (x, y, w, h)
        self.entropy = nn.CrossEntropyLoss()  # Cross-Entropy Loss for class predictions (for multi-class classification)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to normalize predicted box center coordinates to [0, 1]
        
        # Loss function weighting factors
        self.lambda_class = 1  # Weight for class prediction loss
        self.lambda_noobj = 10  # Weight for no-object loss (penalizing false positives)
        self.lambda_obj = 1  # Weight for objectness loss (penalizing false negatives)
        self.lambda_box = 10  # Weight for box coordinate loss (accuracy of bounding boxes)
        
    def forward(self, predictions, target, anchors):
        # Mask to select locations where an object is present (target[..., 0] == 1)
        obj = target[..., 0] == 1
        
        # Mask to select locations where no object is present (target[..., 0] == 0)
        noobj = target[..., 0] == 0
        
        # !No-object loss: Compute Binary Cross-Entropy loss where no object is present
        noobj_loss = self.bce(
            (predictions[..., 0:1][noobj]),  # Predicted objectness score for no-object locations
            (target[..., 0:1][noobj])  # Target objectness score (should be 0 for no-object locations)
        )
        
        # !Object loss (using IoU):
        # Reshape anchors to match the prediction shape for broadcasting during element-wise multiplication
        anchors = anchors.reshape(1, 3, 1, 1, 2)  # Reshape (3, 2) -> (1, 3, 1, 1, 2)
        
        # Compute predicted bounding box coordinates
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]),  # Apply sigmoid to normalize box center (x, y) to [0, 1]
             torch.exp(predictions[..., 3:5]) * anchors],  # Scale width and height using anchor dimensions
            dim=-1  # Concatenate along the last dimension to get final box predictions (x, y, w, h)
        )
        
        # Compute Intersection over Union (IoU) between predicted boxes and ground truth boxes for object locations
        ious = intersection_over_union(
            box_preds[obj],  # Predicted boxes for locations where an object is present
            target[..., 1:5][obj]  # Ground truth boxes for those same locations
        ).detach()
        object_loss = self.bce((predictions[...,0:1][obj]), (ious * target[...,0:1][obj]))
        
        # !Box loss
        predictions[...,1:3] = self.sigmoid(predictions[...,1:3]) # Apply sigmoid to normalize box center (x, y) to [0, 1]
        target[...,3:5] = torch.log(1e-6 + target[...,3:5] / anchors) # this reverses torch.exp(predictions[..., 3:5]) * anchors] to obtain the bbox ground truth anchor offsets
        
        box_loss = self.mse(predictions[...,1:5][obj], target[...,1:5][obj])
        
        
        # !Class loss
        cls_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss  
            + self.lambda_noobj * noobj_loss 
            + self.lambda_class * cls_loss
        )
        
        