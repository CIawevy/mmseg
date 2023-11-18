# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS



@METRICS.register_module()
class MyFmeasure(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 need_noise: bool =False,
                 need_conf: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir_list=[]
        self.output_dir = output_dir
        self.need_noise = need_noise
        self.need_conf = need_conf
        if self.output_dir and is_main_process():
            self.output_dir_list.append(osp.join(self.output_dir,'seg_pred'))
            if self.need_conf:
                self.output_dir_list.append(osp.join(self.output_dir,'conf_map'))
            if self.need_noise:
                self.output_dir_list.append(osp.join(self.output_dir, 'noiseprint'))
            for path in self.output_dir_list:
                mkdir_or_exist(path)
        self.format_only = format_only


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            if data_sample['scores']['data'] is not None:
                pred_det = torch.tensor(data_sample['scores']['data'],device=pred_label.device)
            else:
                pred_det=None
            if data_sample['confidence_map']['data'] is not None:
                confidence_map = data_sample['confidence_map']['data'].squeeze()
            else:
                confidence_map = None

            noiseprint = data_sample['noiseprint']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label).detach().cpu().numpy()
                if pred_det is not None:
                    det_cls = torch.tensor(data_sample['global_class'],device=pred_det.device).detach().cpu().numpy()
                else:
                    det_cls = None
                # self.results.append(
                #     self.intersect_and_union(pred_label, label, num_classes,
                #                              self.ignore_index))
                c_pred_label = pred_label.detach().cpu().numpy()
                c_pred_det = pred_det.detach().cpu().numpy() if pred_det is not None else None
                self.results.append(self.trufor(c_pred_label,
                                                c_pred_det,
                                                label,
                                                det_cls,)
                                                )



            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                for name_path in self.output_dir_list:
                    name = osp.basename(name_path)[0]
                    if name == 's':
                        png_filename = osp.abspath(
                            osp.join(name_path, f'{basename}.png'))
                    elif  name =='n':
                        nop_filename = osp.abspath(
                            osp.join(name_path, f'{basename}.png'))
                    else:
                        conf_filename=osp.abspath(
                            osp.join(name_path, f'{basename}.png'))


                output_mask = pred_label.cpu().numpy()
                if self.need_conf:
                    conf_map = confidence_map.cpu().numpy() *255 #存图待补充 进一步构建文件夹和确认*255 然后输出
                    conf_out = Image.fromarray(conf_map.astype(np.uint8))
                    conf_out.save(conf_filename)

                if self.need_noise:
                    noise = noiseprint.cpu().numpy()
                    noise = (noise - noise.mean()) / (noise.max() - noise.min())*255
                    noise = noise*255
                    ##用mean来做 来开与平均像素的距离更有对比度
                    #标准的npp应该是modal_x = (modal_x - modal_x.mean()) / (modal_x.max() - modal_x.min())
                    # 将标准化后的图像转换为整数类型（0到255之间）
                    integer_noise_out = Image.fromarray(noise.astype(np.uint8))
                    integer_noise_out.save(nop_filename)
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                #注意这里 还需要加存图
                output_mask = (output_mask > 0.5).astype(np.uint8)*255
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])

        # self.results 这里修改 四个results 的 np.nanmean 和 pretty 输出
        results = tuple(zip(*self.results))
        F1_b , F1_t = results[0] , results[1]
        det_pred , det_cls = results[2] , results[3]
        F1_best , F1_th = self.computeFscore(F1_b,F1_t,self.loc_met_cfg)
        if det_pred[0] is not None:
            AUC , ACC = self.computeDetectionMetrics(det_pred , det_cls , self.det_met_cfg)
        else:
            AUC , ACC = None, None
        results = [F1_best , F1_th , AUC , ACC]
        ret_metrics = self.counting_metrics(
            self.metrics, self.nan_to_num, results )

        # class_names = self.dataset_meta['classes']


        metrics = dict()


        # each class table

        ret_metrics_class = OrderedDict({
            ret_metric: [np.round(ret_metric_value * 100, 2)]
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # ret_metrics_class.update({'Class': class_names})
        # ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)






        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
    def computeFscore(self,fb , ft, cfg ):
        f_th = np.nanmean(ft) if cfg[1] else None
        f_b = np.nanmean(fb) if cfg[0] else None
        return f_b , f_th
    def trufor(self,pred, pred_det, gt, det_cls ):
        self._calculating_cfg()
        F1_best, F1_th = self.computeLocalizationMetrics(pred , gt , self.loc_met_cfg)

        return  F1_best , F1_th , pred_det , det_cls
    def _calculating_cfg(self):
        localization_cfg = [False, False]
        detection_cfg = [False, False]
        met = self.metrics
        for m in met:
            if m == 'F1_th':
                localization_cfg[1] = True
            elif m == 'F1_best':
                localization_cfg[0] = True
            elif m == 'ACC' :
                detection_cfg[1] = True
            elif m == 'AUC':
                detection_cfg[0]=True
        self.loc_met_cfg = localization_cfg
        self.det_met_cfg = detection_cfg
    def computeDetectionMetrics(self,scores, labels , det_cfg:list):
        lbl = np.array(labels)
        lbl = lbl[np.isfinite(scores)]

        scores = np.array(scores, dtype='float32')
        scores[scores == np.PINF] = np.nanmax(scores[scores < np.PINF])
        scores = scores[np.isfinite(scores)]
        assert lbl.shape == scores.shape

        # AUC

        from sklearn.metrics import roc_auc_score
        AUC = roc_auc_score(lbl, scores) if det_cfg[0] else None

        # Balanced Accuracy
        from sklearn.metrics import balanced_accuracy_score
        bACC = balanced_accuracy_score(lbl, scores > 0.5) if det_cfg[1] else None

        return AUC, bACC



    def computeLocalizationMetrics(self,pred: torch.tensor, gt: torch.tensor, loc_cfg:list,
                            ):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
            from scipy.ndimage.filters import minimum_filter, maximum_filter
            gt1 = minimum_filter(gt, erodeKernSize)
            gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
            return gt0, gt1
            # gtlist=[gt0,gt1,gt0+gt1]
            # # format_result
            # for index,output_mask in enumerate(gtlist):
            #     if self.output_dir is not None:
            #         name=['min','max','mix'][index]
            #         basename = osp.splitext(osp.basename(
            #             data_sample['img_path']))[0]
            #         png_filename = osp.abspath(
            #             osp.join(self.output_dir,name, f'{basename}.png'))
            #         if not osp.exists(osp.join(self.output_dir, name)):
            #             os.mkdir(osp.join(self.output_dir, name))
            #     # The index range of official ADE20k dataset is from 0 to 150.
            #     # But the index range of output is from 0 to 149.
            #     # That is because we set reduce_zero_label=True.
            #
            #     #注意这里
            #         output_mask = output_mask * 255
            #         output = Image.fromarray(output_mask.astype(np.uint8))
            #         output.save(png_filename)
            # return gt0,gt1

        def computeMetricsContinue(values, gt0, gt1):
            values = values.flatten().astype(np.float32)
            gt0 = gt0.flatten().astype(np.float32)
            gt1 = gt1.flatten().astype(np.float32)

            inds = np.argsort(values)
            inds = inds[(gt0[inds] + gt1[inds]) > 0]
            vet_th = values[inds]
            gt0 = gt0[inds]
            gt1 = gt1[inds]

            TN = np.cumsum(gt0)
            FN = np.cumsum(gt1)
            FP = np.sum(gt0) - TN
            TP = np.sum(gt1) - FN

            msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
            FP = FP[msk]
            TP = TP[msk]
            FN = FN[msk]
            TN = TN[msk]
            vet_th = vet_th[msk]

            return FP, TP, FN, TN, vet_th

        def computeMetrics_th(values, gt, gt0, gt1, th):
            values = values > th
            values = values.flatten().astype(np.uint8)
            gt = gt.flatten().astype(np.uint8)
            gt0 = gt0.flatten().astype(np.uint8)
            gt1 = gt1.flatten().astype(np.uint8)

            gt = gt[(gt0 + gt1) > 0]
            values = values[(gt0 + gt1) > 0]

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(gt, values)

            TN = cm[0, 0]
            FN = cm[1, 0]
            FP = cm[0, 1]
            TP = cm[1, 1]

            return FP, TP, FN, TN

        def computeF1(FP, TP, FN, TN):
            return 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)

        gt0, gt1 = extractGTs(gt)

        if loc_cfg[0]:
            try:
                FP, TP, FN, TN, _ = computeMetricsContinue(pred, gt0, gt1)
                f1 = computeF1(FP, TP, FN, TN)
                f1i = computeF1(TN, FN, TP, FP)
                F1_best = max(np.max(f1), np.max(f1i))
            except:
                import traceback
                traceback.print_exc()
                F1_best = np.nan

                # fixed threshold
        else:
            F1_best=None
        if  loc_cfg[1]:
            try:
                FP, TP, FN, TN = computeMetrics_th(pred, gt, gt0, gt1, 0.5)
                f1 = computeF1(FP, TP, FN, TN)
                f1i = computeF1(TN, FN, TP, FP)
                F1_th = max(f1, f1i)
            except:
                import traceback
                traceback.print_exc()
                F1_th = np.nan
        else:
            F1_th=None

        return F1_best, F1_th




    @staticmethod
    def counting_metrics(   metrics: List[str] = ['F1_th'],
                            nan_to_num: Optional[int] = None,
                            results=None):
        """Calculate evaluation metrics
        Args:
            results: dict
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """


        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = [ 'F1_best', 'F1_th', 'AUC' , 'ACC' ]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        ret_metrics = OrderedDict()
        for metric in metrics:
            if metric == 'F1_th':
                ret_metrics['F1_th'] = results[1]
            elif metric == 'F1_best':
                ret_metrics['F1_best'] = results[0]
            elif metric == 'AUC':
                ret_metrics['AUC'] = results[2]
            else:
                ret_metrics['ACC'] = results[3]


        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
