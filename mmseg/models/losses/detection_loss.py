import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weighted_loss
import warnings
from .utils import get_class_weight, weight_reduce_loss




def balanced_cross_entropy(pred,
                         label,
                         reduction='mean',
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`

    Returns:
        torch.Tensor: The calculated loss"""
    if pred.shape != label.shape:
        pred = pred.squeeze()
    # 计算正类别和负类别的样本数
    num_pos = label.sum()
    num_neg = label.size(0) - num_pos
    # 计算平衡的正类别和负类别权重
    balance_weight = num_neg / (num_pos + 1e-5)

    balance_weight=torch.tensor([balance_weight],device=pred.device)#正类别损失
    label.to(pred.deice)
    #class_weight Tensor([n]) n为正样本数量
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=balance_weight, reduction=reduction)



    return loss



@MODELS.register_module()
class DetectionLoss(nn.Module):

    def __init__(self, reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_det',
                 loss_sta='detection',
                 avg_non_ignore=False,
                 ):
        super(DetectionLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.loss_sta=loss_sta
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                **kwargs):


        loss = self.loss_weight * balanced_cross_entropy(
            cls_score,
            label,
            **kwargs)


        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

