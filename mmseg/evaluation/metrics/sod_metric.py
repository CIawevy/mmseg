# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
from py_sod_metrics.utils import EPS, TYPE, get_adaptive_threshold, prepare_data
from collections import OrderedDict
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
import py_sod_metrics
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.registry import METRICS

@METRICS.register_module()
class SodMetric(BaseMetric):
    """Sod evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        sod_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'MAE', 'S-measure','E-measure' and 'wF-measure'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (float):  the weight of the precision in wF-measure
            Default: 1.0.
        alpha (float) : the weight for balancing the object score and the region score in S-measure
            Default: 0.5
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 sod_metrics: List[str] = ['MAE'],
                 nan_to_num: Optional[int] = None,
                 alpha:float = 0.5,
                 beta: float = 1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = sod_metrics
        self.nan_to_num = nan_to_num
        self.alpha=alpha
        self.beta=beta
        #To use methods bellow ,I have to initialize the sod_metirc  class first ,already import
        # cFM = py_sod_metrics.Fmeasure()
        self.cWFM = py_sod_metrics.WeightedFmeasure()
        self.cSM = py_sod_metrics.Smeasure()
        self.cEM = py_sod_metrics.Emeasure()
        self.cMAE = py_sod_metrics.MAE()


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
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            # if 'S-measure' not in self.metrics and 'E-measure' not in self.metrics and 'wF-measure' not in self.metrics:
            #     mask = (label != self.ignore_index)
            #     pred_label = pred_label[mask]
            #     label = label[mask]
            pred, gt = prepare_data(pred_label, label*255)
            """
            it is necessary to note that ignore_index is not appropriate for E-measure or S-measure 
            due to the original principle of the evaluation,which need gt to be of two dimensions shape
            which must be as same as the rgb image shape
            while it can still be used ,when only counting for MAE or f-measure
            """
            allowed_metrics = ['MAE', 'S-measure', 'E-measure', 'wF-measure']
            if not set(self.metrics).issubset(set(allowed_metrics)):
                raise KeyError(f'metrics {self.metrics} is not supported')
            metric_list=[]
            for me in self.metrics:
                if me=='MAE':
                    mae = self.cMAE.cal_mae(pred, gt)
                    metric_list.append(mae)
                elif me =='S-measure':
                    sm = self.cSM.cal_sm(pred, gt)
                    metric_list.append(sm)
                elif me=='E-measure':
                    self.cEM.gt_fg_numel = np.count_nonzero(gt)  # for cem method usage
                    self.cEM.gt_size = gt.shape[0] * gt.shape[1]  # for cem method usage
                    changeable_ems = self.cEM.cal_changeable_em(pred, gt)
                    changeable_em = np.mean(np.array(changeable_ems, dtype=TYPE), axis=0)#如果想获得maxE，minE就不要合并这一步，这样会有一个[val,val]的数组返回
                    # adaptive_em = self.cEM.cal_adaptive_em(pred, gt)

                    # metric_list.append(adaptive_em)
                    metric_list.append(changeable_em)

                elif me=='wF-measure':
                    if np.all(~gt):
                        wfm = 0
                        metric_list.append(wfm)
                    else:
                        wfm = self.cWFM.cal_wfm(pred, gt)
                        metric_list.append(wfm)


            self.results.append(metric_list)
        if not len(self.results)%10:
            print(f'results at ={len(self.results)}/6473')
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        values can be found from self.xxx usage
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes MAE, WeighedF-measure, S-measure, mean E-measure, ...
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        results=tuple(zip(*results))
        rets=self.Total_metrics(
            results, self.metrics, self.nan_to_num)
        #get results
        class_names=tuple(['value'])

        metrics = dict()
        for key, val in rets.items():
            if key == 'MAE':
                metrics[key] = val
            else:
                metrics['mean' + key] = val

        # each class table
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value,3)
            for ret_metric, ret_metric_value in rets.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics



    @staticmethod
    def Total_metrics(results: list,
                      metrics:List[str] = ['MAE'],
                      nan_to_num: Optional[int] = None):
        """Calculate evaluation metrics
        Args:
            metrics (List[str] | str): Metrics to be evaluated, 'MAE' and
                'wF-measure','S-measure','E-measure'
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['MAE', 'S-measure','E-measure' ,'wF-measure']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')
        index=0
        ret_metrics = OrderedDict()
        for metric in metrics:
            if metric == 'MAE':
                maes=results[index]
                mae = np.mean(np.array(maes, TYPE))
                ret_metrics[metric] = [mae]
                index+=1
            elif metric == 'S-measure' :
                sms=results[index]
                sm = np.mean(np.array(sms, dtype=TYPE))
                ret_metrics[metric] = [sm]
                index+=1
            elif metric == 'E-measure' :
                adaptive_ems=results[index]
                changeable_ems=results[index]
                adaptive_em = np.mean(np.array(adaptive_ems, dtype=TYPE))
                changeable_em = np.mean(np.array(changeable_ems, dtype=TYPE), axis=0)#获得meanE
                ret_metrics['adaptive_em'] = [adaptive_em]
                ret_metrics['changeable_em']=[changeable_em]
                index+=1
            else:
                weighted_fms=results[index]
                weighted_fm = np.mean(np.array(weighted_fms, dtype=TYPE))
                ret_metrics[metric]=[weighted_fm]
                index+=1

        if nan_to_num is not None:
            ret_metrics = OrderedDict[{
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            }]
        return ret_metrics

