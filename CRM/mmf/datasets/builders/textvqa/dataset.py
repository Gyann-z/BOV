# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.text import word_tokenize

import math
from PIL import Image
from torchvision import transforms
from mmf.common.sample import to_device
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

import fasttext

class TextVQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("textvqa", config, dataset_type, index=imdb_file_index)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

    def preprocess_sample_info(self, sample_info):
        path = self._get_path_based_on_index(self.config, "annotations", self._index)
        # NOTE, TODO: Code duplication w.r.t to STVQA, revisit
        # during dataset refactor to support variable dataset classes
        if "stvqa" in path:
            feature_path = sample_info["feature_path"]
            append = "train"

            if self.dataset_type == "test":
                append = "test_task3"

            if not feature_path.startswith(append):
                feature_path = append + "/" + feature_path

            sample_info["feature_path"] = feature_path
            return sample_info
        # COCO Annotation DBs have corrext feature_path
        #change
        elif "COCO" not in sample_info["feature_path"]:
            sample_info["feature_path"] = sample_info["image_path"].replace(
                ".jpg", ".npy"
            )
        return sample_info

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_ids = report.image_id.cpu().numpy()
        context_tokens = report.context_tokens.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                    pred_source.append("OCR")
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]
        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead

        if hasattr(current_sample, "image_info_0"):
            for k in list(current_sample.image_info_0):
                if k != "max_features":
                    current_sample.image_info_0.pop(k)
        if hasattr(current_sample, "image_info_1"):
            for k in list(current_sample.image_info_1):
                if k != "max_features":
                    current_sample.image_info_1.pop(k)

        return current_sample

    def bb_intersection_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        assert (boxBArea + boxAArea - interArea) != 0

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def add_sample_details(self, sample_info, sample):

        sample.image_id = object_to_byte_tensor(sample.image_id)

        # 1. Load text (question words)
        question_str = (
            sample_info["question"]
            if "question" in sample_info
            else sample_info["question_str"]
        )
        text_processor_args = {"text": question_str}

        if "question_tokens" in sample_info:
            text_processor_args["tokens"] = sample_info["question_tokens"]

        processed_question = self.text_processor(text_processor_args)

        if "input_ids" in processed_question:
            sample.text = processed_question["input_ids"]
            sample.text_len = torch.tensor(
                len(processed_question["tokens"]), dtype=torch.long
            )
        else:
            # For GLoVe based processors
            sample.text = processed_question["text"]
            sample.text_len = processed_question["length"]

        # 2. Load object
        # object bounding box information
        if "obj_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
            sample.obj_bbox_coordinates = self.copy_processor(
                {"blob": sample_info["obj_normalized_boxes"]}
            )["blob"]

        bbA = sample_info['ocr_normalized_boxes'][sample_info['token_idx']]
        bbox = sample_info["obj_normalized_boxes"]
        answer_relation_matrix = np.zeros((100))  # (100)
        xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)  # (100,1),(100,1),(100,1),(100,1)
        for b in range(100):
            bbB = bbox[b]
            # (YK): Padded bbox
            if sum(bbB) == 0:
                continue
                # class 1: inside (i inside token)
            if (
                bbA[0] < xmin[b]
                and bbA[2] > xmax[b]
                and bbA[1] < ymin[b]
                and bbA[3] > ymax[b]
            ):
                answer_relation_matrix[b] = 1

            # class 2: cover (i covers token)
            elif (
                xmin[b] < bbA[0]
                and xmax[b] > bbA[2]
                and ymin[b] < bbA[1]
                and ymax[b] > bbA[3]
            ):
                answer_relation_matrix[b] = 1
            else:
                ioU = self.bb_intersection_over_union(bbA, bbB)

                # class 3: i and j overlap
                if ioU >= 0.3:  # change 0.5
                    answer_relation_matrix[b] = 1

        # change
        rel_obj=answer_relation_matrix
        obj_bbox_rel = torch.full((10, 4), fill_value=0, dtype=torch.float)  # (10,2048)
        obj_bbox_filter = []
        for obj_i in range(len(rel_obj)):
            if rel_obj[obj_i] == 1:
                obj_bbox_filter.append(sample.obj_bbox_coordinates[obj_i])

        length = min(len(obj_bbox_filter), 10)

        for idx in range(length):
            obj_bbox_rel[idx] = obj_bbox_filter[idx]

        sample.obj_bbox_coordinates=obj_bbox_rel

        obj_fc6_rel = torch.full((10, 2048), fill_value=0, dtype=torch.float)  # (10,2048)
        obj_fc6_filter = []
        for obj_i in range(len(rel_obj)):
            if rel_obj[obj_i] == 1:
                obj_fc6_filter.append(sample.image_feature_0[obj_i])

        length = min(len(obj_fc6_filter), 10)

        for idx in range(length):
            obj_fc6_rel[idx] = obj_fc6_filter[idx]

        sample.image_feature_0=obj_fc6_rel #(10,2048)

        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            if "image_feature_1" in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            return sample

        # Preprocess OCR tokens
        if hasattr(self, "ocr_token_processor"):
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
        else:
            ocr_tokens = sample_info["ocr_tokens"]
        # Get FastText embeddings for OCR tokens

        ocr_tvs_feats = torch.tensor(sample.image_feature_2)
        sample.context_encoder_feats = ocr_tvs_feats

        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.ocr_tokens = context["tokens"]

        sample.context_tokens = object_to_byte_tensor(context["tokens"])
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})
            sample.context_feature_1 = context_phoc["text"]
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]

        # OCR order vectors
        if self.config.get("use_order_vectors", False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        # OCR bounding box information
        if "ocr_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": sample_info["ocr_normalized_boxes"]}
            )["blob"][:max_len]
        elif self.use_ocr_info and "ocr_info" in sample_info:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates


        # change
        # load self
        token_idx = sample_info["token_idx"]
        sample.self_token=sample.ocr_tokens[token_idx]
        sample.self_context = sample.context[token_idx]
        sample.self_context_tokens = object_to_byte_tensor(sample.self_token)
        sample.self_context_feature_1 = sample.context_feature_1[token_idx]
        sample.self_order_vectors = sample.order_vectors[token_idx]
        sample.self_ocr_bbox_coordinates=sample.ocr_bbox_coordinates[token_idx]
        sample.self_context_encoder_feats=ocr_tvs_feats[token_idx]
        sample.self_image_feature_1=sample.image_feature_1[token_idx]
        sample.self_context_info_0 = Sample()
        sample.self_context_info_0.max_features = torch.tensor(1)

        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answer = sample_info.get("answer")
        related_label=sample_info.get("related_label")
        answer_processor_arg = {"answers": answers,"answer":answer,"related_label":related_label}#change

        answer_processor_arg["tokens"] = sample.pop("ocr_tokens", [])

        processed_answers = self.answer_processor(answer_processor_arg)
        #dict,{'answers':len=10,'answers_scores','sampled_idx_seq','train_prev_inds','label':(100,),'label_len','answer'}

        assert not self.config.fast_read, (
            "In TextVQADataset, online OCR sampling is incompatible "
            "with fast_read, so fast_read is currently not supported."
        )

        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if "answers_scores" in sample:
            sample.targets = sample.pop("answers_scores")

        return sample

    #change
    class ResizeNormalize(object):
        def __init__(self, size, interpolation=Image.BILINEAR):
            self.size = size
            self.interpolation = interpolation
            self.toTensor = transforms.ToTensor()

        def __call__(self, img):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img
