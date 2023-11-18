import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weighted_loss
import warnings
def conf_loss(pred,#b,c,H,W
              conf,#B,1,H,W
              target,#B,H,W
              reduction='mean',
              pad_list=None, #B 4
                ):
    #ti = (1 − gi) (1 − ai) + gi ai #Tensor 大部分损失函数是自带 sigmoid 和 softmax的
    b, _ ,H , W = pred.shape[0] ,pred.shape[1] , pred.shape[2], pred.shape[3]
    padding_info=pad_list
    if pad_list is not None:
        condition = torch.full((b, 1, H, W), bool(False), device=pred.device)
        for i in range(pad_list.shape[0]):
            padding_left, padding_right, padding_top, padding_bottom = \
                padding_info[i].int()
            condition[i, :, padding_top:(H - padding_bottom), padding_left:(W - padding_right)] = True

    condition = condition.squeeze().view(b,-1)  #B H*W

    new_target = target.squeeze().view(b,-1)[condition]
    conf = conf.squeeze() # B H W
    new_conf = torch.sigmoid(conf).view(b,-1)[condition]
    new_pred = F.softmax(pred, dim=1)[:,1,:,:].view(b,-1)[condition]


    reference=(1-new_target)*(1-new_pred)+new_pred*new_target
    loss=F.mse_loss(new_conf, reference, reduction=reduction )
    return loss


@MODELS.register_module()
class ConfLoss(nn.Module):

    def __init__(self, reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_conf',
                 loss_sta='confidence',
                 avg_non_ignore=False,
                 ):
        super(ConfLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.loss_sta=loss_sta
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain target and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')
        self._loss_name = loss_name

    def forward(self,
                pred,
                conf,
                target,
                pad,
                reduction_override=None,):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * conf_loss(
            pred,
            conf,
            target,
            pad_list=pad,
            reduction=reduction,)

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


