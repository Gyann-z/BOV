# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
import string
import math
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import is_master

from mmf.modules.labelmaps import get_vocabulary
import time
logger = logging.getLogger(__name__)


#change
def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()

def to_numpy(tensor):
  if torch.is_tensor(tensor):
    return tensor.cpu().numpy()
  elif type(tensor).__module__ != 'numpy':
    raise ValueError("Cannot convert {} to numpy array"
                     .format(type(tensor)))
  return tensor

# change
voc = get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')  # list,len=97
char2id = dict(zip(voc, range(len(voc))))  # dict,len=97
id2char = dict(zip(range(len(voc)), voc))  # dict,len=97
eos=char2id['EOS']

def get_str_list(output, target):
  # label_seq
  assert output.dim() == 2 and target.dim() == 2

  end_label = char2id['EOS']
  unknown_label = char2id['UNKNOWN']
  num_samples, max_len_labels = output.size()
  num_classes = len(char2id.keys())
  assert num_samples == target.size(0) and max_len_labels == target.size(1)
  output = to_numpy(output)
  target = to_numpy(target)

  # list of char list
  pred_list, targ_list = [], []
  for i in range(num_samples):
    pred_list_i = []
    for j in range(max_len_labels):
      if output[i, j] != end_label:
        if output[i, j] != unknown_label:
          pred_list_i.append(id2char[output[i, j]])
      else:
        break
    pred_list.append(pred_list_i)

  for i in range(num_samples):
    targ_list_i = []
    for j in range(max_len_labels):
      if target[i, j] != end_label:
        if target[i, j] != unknown_label:
          targ_list_i.append(id2char[target[i, j]])
      else:
        break
    targ_list.append(targ_list_i)

  # char list to string
  # if dataset.lowercase:
  if True:
    # pred_list = [''.join(pred).lower() for pred in pred_list]
    # targ_list = [''.join(targ).lower() for targ in targ_list]
    pred_list = [_normalize_text(pred) for pred in pred_list]
    targ_list = [_normalize_text(targ) for targ in targ_list]
  else:
    pred_list = [''.join(pred) for pred in pred_list]
    targ_list = [''.join(targ) for targ in targ_list]

  return pred_list, targ_list

def Accuracy(output, target, dataset=None):#output(3371,1),target(3371)
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    # prediction = torch.argmax(output, 1)
    prediction = output.ge(0.5).float()
    correct = (prediction == target.unsqueeze(-1)).sum().float()
    total += len(target)
    Accuracy= correct / total
    return Accuracy

  # pred_list, targ_list = get_str_list(output, target)
  #
  # acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
  # accuracy = 1.0 * sum(acc_list) / len(acc_list)
  # return accuracy

def RecPostProcess(output, target, score, dataset=None):
  pred_list, targ_list = get_str_list(output, target)
  max_len_labels = output.size(1)
  score_list = []

  score = to_numpy(score)
  for i, pred in enumerate(pred_list):
    len_pred = len(pred) + 1 # eos should be included
    len_pred = min(max_len_labels, len_pred) # maybe the predicted string don't include a eos.
    score_i = score[i,:len_pred]
    score_i = math.exp(sum(map(math.log, score_i)))
    score_list.append(score_i)

  return pred_list, targ_list, score_list


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            targets=[]
            outputs={}

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):

                report = self._forward(batch)
                targets.append(torch.tensor(report.related_label))

                for k, v in report.items():
                    if k in ['pred_related_score']:
                        if k not in outputs:
                            outputs[k] = []
                        outputs[k].append(v.cpu())

                self.update_meter(report, meter)


                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            targets = torch.cat(targets)
            for k, v in outputs.items():
                outputs[k] = torch.cat(outputs[k])

            eval_res = Accuracy(outputs['pred_related_score'], targets)
            print('lexicon0: {0}: {1:.3f}'.format('accuarcy', eval_res))
            # pred_list, targ_list, score_list = RecPostProcess(outputs['pred_rec'], targets, outputs['pred_rec_score'])
            # with open("pre_results_"+str(time.time())+".txt", "w", encoding="utf-8") as f:
            #     for pred, targ in zip(pred_list, targ_list):
            #         f.write("{} {}\n".format(pred, targ))

            # combined_report.metrics = self.metrics(combined_report, combined_report)
            # self.update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            logger.info("Finished predicting")
            self.model.train()
