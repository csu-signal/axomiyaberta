
#Author: Abhijnan Nath, CSU Signal Lab. Extended from https://github.com/AI4Bharat/indic-bert

import torch.nn as nn
import torch
from transformers import *
from transformers import AutoModel, AutoTokenizer, AutoModelForMultipleChoice, AutoModelForTokenClassification

 
import numpy as np
import pyhocon
import os
from inspect import getfullargspec

import csv
import json
import os

import tqdm
import logging

from dataclasses import dataclass
from typing import Optional, List, Any, Union
from transformers import PreTrainedTokenizer
from collections import defaultdict

logger = logging.getLogger(__name__)

model_name="/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500"
from transformers import (AutoModelWithLMHead, 
                          AutoTokenizer, 
                          BertConfig)
device = torch.device('cuda:0')
#albert_model = AutoModelForMultipleChoice.from_pretrained(model_name) 
albert_model = AlbertModel.from_pretrained(model_name).to(device)

tokenizer = AlbertTokenizer.from_pretrained("/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/")
tokenizer.model_max_length = 514


import os
import sys
import pickle
import gc
import numpy as np
 # for reproducibility 
import csv
gc.collect()
import torch
import torch.nn as nn
import torch.nn as F
from collections import defaultdict
import matplotlib.pyplot as plt 
#torch.cuda.empty_cache()
 
print(torch.cuda.current_device())
#parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')

parent_path = '/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/'

sys.path.append(parent_path)

import os.path
import pickle

from sklearn.model_selection import train_test_split
import pyhocon
from wiki_models import AxBERTa_Classfier
 
import random
#from tqdm.autonotebook import tqdm
from parsing.parse_ecb import parse_annotations


import csv
import json
import os

import logging

from dataclasses import dataclass
from typing import Optional, List, Any, Union
from transformers import PreTrainedTokenizer
logger = logging.getLogger(__name__)
from collections import defaultdict

from examples import MultipleChoiceExample, TextExample, TokensExample
 
@dataclass(frozen=True)
class MultipleChoiceExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence
        (question).
        contexts: list of str. The untokenized text of the first sequence
        (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be
        equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]

class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, lang, mode):
        if mode == 'train':
            return self.get_train_examples(lang)
        elif mode == 'dev':
            return self.get_dev_examples(lang)
        elif mode == 'test':
            return self.get_test_examples(lang)

    def modes(self):
        return ['train', 'dev', 'test']

    def get_train_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self, lang):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, encoding='utf-8') as fp:
            return list(csv.reader(fp, delimiter=','))

    @classmethod
    def read_json(cls, input_file):
        """Reads a json file file."""
        with open(input_file, encoding='utf-8') as fp:
            return json.load(fp)

    @classmethod
    def readlines(cls, filepath):
        with open(filepath, encoding='utf-8') as fp:
            return fp.readlines()

    @classmethod
    def read_jsonl(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as fp:
            data = fp.readlines()
            data = list(map(lambda l: json.loads(l), data))
        return data

class WikiSectionTitles(DataProcessor):
    """Processor for the Wikipedia Section Title Prediction dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/{}-train.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/{}-valid.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'dev')

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}/{}-test.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        return ['titleA', 'titleB', 'titleC', 'titleD']

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = [
            MultipleChoiceExample(
                example_id=idx,
                question='',contexts=[item['sectionText'], item['sectionText'],
                          item['sectionText'], item['sectionText']],
                
                
#                 contexts=['<m>'+ " " + item['sectionText']+ " "+'</m>', '<m>'+ " " + item['sectionText']+ " "+'</m>',
#                           '<m>'+ " " + item['sectionText']+ " "+'</m>', '<m>'+ " " +item['sectionText']+ " "+'</m>'],
                
                endings=['<m>'+ " " + item['titleA']+ " "+'</m>','<m>'+" "+item['titleB']+" "+'</m>','<m>'+" "+item['titleC']+" "+'</m>',
                         '<m>'+" "+item['titleD']+" "+'</m>'],
                label=item['correctTitle'],
            )
            for idx, item in enumerate(items)
        ]
        return examples
    
 


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: Any
    attention_mask: Any
    token_type_ids: Any = None
    label: Any = None
    candidates: Any = None
    example_id: str = None


def convert_multiple_choice_examples_to_features(
    examples: List[MultipleChoiceExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    label_list: List[str],
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
#     max_length= 1024
    features_dict = defaultdict(dict)
    count = 0
    label_map = {label: i for i, label in enumerate(label_list)}
    id2_label_map = {i: label for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context 
            #print(count, ending_idx)
            if example.question.find("_") != -1:
                print("no question cannot find")
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                #text_b = text_a + " " + ending
                text_b =  '<m>' +'</m>' + ending
                
                

            inputs = tokenizer(text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation='only_first',
                pad_to_max_length=True,return_tensors = 'pt'
            )
            
#             inputs = tokenizer(
#                 text_b,
#                 add_special_tokens=False,
#                 max_length=max_length,
#                 truncation=False,
#                 pad_to_max_length=True,return_tensors = 'pt'
#             )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)
            
            if count<=20000:
                #flatten all the examples for getting the features of each of the 4 contexts inside a sample
                #since we are feeing each individual sample with their label into the classifier. bin classifier!

                #print("count", count)
       
                
 
                features_dict[count]["input_ids"] = inputs["input_ids"]
                features_dict[count]["attention_mask"] = inputs["attention_mask"]
                features_dict[count]["token_type_ids"] = torch.zeros(max_length)
                
                features_dict[count]["position_ids"] = torch.arange(max_length)
                #features_dict[count]["position_ids"] = torch.arange(inputs["input_ids"].shape[-1]).expand(inputs["input_ids"].shape)
                features_dict[count]["label"] = 1 if ending_idx == label_map[example.label] else 0
                count+=1

                #upsampling positive examples from the dev set because of the class imbalance 
            
            
            elif count>20000:
                label= (ending_idx == label_map[example.label])
                if label == 1:
                    
                    print("dev positive")
                    
                    
                    features_dict[count]["input_ids"] = inputs["input_ids"] 
                    features_dict[count]["attention_mask"] = inputs["attention_mask"]
                    features_dict[count]["token_type_ids"] = torch.zeros(max_length)
                    features_dict[count]["position_ids"] =torch.arange(max_length)
                    #features_dict[count]["position_ids"] = torch.arange(inputs["input_ids"].shape[-1]).expand(inputs["input_ids"].shape)
                    features_dict[count]["label"] = 1 if ending_idx == label_map[example.label] else 0
                    count+=1
                    

            
            
        label = label_map[example.label]
       

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )
       
      
    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features_dict, label_map


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(predicted_labels)


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(true_labels)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    return 2 * P * R / (P + R)




def print_label_distri(labels):
    label_count = {}
    for label in labels:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    print(len(labels))
    label_count_ratio = {label: val / len(labels) for label, val in label_count.items()}
    return label_count_ratio


def split_data(all_examples, dev_ratio=0.2):
    pairs, labels = zip(*all_examples)
    return train_test_split(pairs, labels, test_size=dev_ratio)


def get_arg_attention_mask(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    input_ids.cpu()

    num_inputs = input_ids.shape[0]
    m = input_ids.cpu()

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id
    
    k = m == parallel_model.module.vals[0]
    p = m == parallel_model.module.vals[1]
    v = (k.int() + p.int()).bool()
    #print("non zero counts", v.nonzero().shape)
    #print(k.shape,p.shape)
    #m = m_start_indicator + m_end_indicator
    #v = (m_start_indicator.int() + m_end_indicator.int()).bool()

    # non-zero indices are the tokens corresponding to <m> and </m>
    #nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))
 
    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)
    

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    
    # I think CLS token should also have global attention apart from the mentions 
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()
    
    
 

    return attention_mask_g, arg1, arg2


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only)

def forward_axberta(parallel_model, train_pairs, device, indices, lm_only=False):
    idx = [train_pairs[i]['input_ids'] for i in train_indices[0:5]]
    idx = [train_pairs[i]['attention_mask'] for i in train_indices[0:5]]
    idx = [train_pairs[i]['position_ids'] for i in train_indices[0:5]]
    #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]
    idx
    a = torch.stack(idx, dim =0).squeeze()
    
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only)
 
def generate_lm_out_wiki(parallel_model, device, dev_ab, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    ab_lm_out_all = []
 
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, n, batch_size), desc="Generating LM Outputs"):
            batch_indices = indices[i: i + batch_size]
            lm_out_ab = forward_ab(parallel_model, dev_ab, device, batch_indices, lm_only=True).detach().cpu()
            ab_lm_out_all.append(lm_out_ab)
 
    return {'ab': torch.vstack(ab_lm_out_all)}


def frozen_predict(parallel_model, device, dev_ab, batch_size, dev_labels,  force_lm_output=False):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    validation_loss= []
    bce_loss = torch.nn.BCELoss()
    lm_out_dict = generate_lm_out_wiki(parallel_model, device, dev_ab, batch_size)
 

    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, n, batch_size), desc="Predicting"):
            batch_indices = indices[i: i + batch_size]
            batch_labels = dev_labels[batch_indices].to(device)
            ab_out = lm_out_dict['ab'][batch_indices, :]
            #ba_out = lm_out_dict['ba'][batch_indices, :]
            ab_out.to(device)
            #ba_out.to(device)
            scores_ab = parallel_model(ab_out, pre_lm_out=True)
            
            val_loss = bce_loss(torch.squeeze(scores_ab), batch_labels)
            #scores_ba = parallel_model(ba_out, pre_lm_out=True)
            #scores_mean = (scores_ab + scores_ba)/2
            print("val scores", scores_ab)
            batch_predictions = (scores_ab > 0.10).detach().cpu()
            print("batch dev predictions",batch_predictions)
            predictions.append(batch_predictions)
            #validation_loss.append(val_loss)

    return torch.cat(predictions), val_loss

def cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    cos_dev = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
     
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            cos_ab = forward_ab(parallel_model, dev_ab, device, batch_indices) # these scores are actually embeddings 
            cos_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            batch_predictions = cos_dev(cos_ab, cos_ba).detach().cpu()
            print("batch dev prediction tensor", batch_predictions)
            batch_predictions = (batch_predictions > 0.2).detach().cpu()
            #print("batch dev prediction boolean", batch_predictions)

         
#             batch_predictions = (cos_sim > 0.7).detach().cpu()
            
#             condition = (batch_predictions>0.987).detach().cpu()
#             print("condition", condition)
#             #print(condition.size())
#             batch_predictions = batch_predictions.where(condition, torch.tensor(0.0))
#             batch_predictions = batch_predictions.where(~condition, torch.tensor(1.0))
# #             batch_predictions = batch_predictions.detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)


#def predict(parallel_model, device, dev_ab, dev_ba, batch_size):
def predict(parallel_model, device, dev_ab, batch_size):

    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab, _ = forward_ab(parallel_model, dev_ab, device, batch_indices)
            print("bce scores", scores_ab)
            #scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)

            #scores_mean = (scores_ab + scores_ba) / 2

            batch_predictions = (scores_ab > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)
def predict_c_ent(parallel_model, device, dev_ab, batch_size):

    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab, _ = forward_ab(parallel_model, dev_ab, device, batch_indices)
            print("bce scores", scores_ab)
            #scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)

            #scores_mean = (scores_ab + scores_ba) / 2
            batch_predictions = scores_ab.argmax(dim=-1)
            #batch_predictions = (scores_ab > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)

def predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size, dev_labels):
    
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()
#     batch_tensor_ab = tokenized_feature_dict_dev['input_ids'][indices, :]
#     batch_am_ab = tokenized_feature_dict_dev['attention_mask'][indices, :]
#     batch_posits_ab = tokenized_feature_dict_dev['position_ids'][indices, :]
    
    n = tokenized_feature_dict_dev['input_ids'].shape[0]
    
    
#     dev_labels = dev_labels.to(device)
    indices = list(range(n))
    predictions = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    parallel_model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            batch_dev_labels = dev_labels[i: i + batch_size].to(device)

            scores_ab = forward_ab(parallel_model, tokenized_feature_dict_dev, device, batch_indices, lm_only=True) 
            scores, scores_ba = forward_ab(parallel_model, tokenized_feature_dict_dev, device, batch_indices)
            
            val_loss = 0.01*(bce_loss(torch.squeeze(scores), batch_dev_labels)) +(cos_loss ( scores_ba,scores_ab, batch_dev_labels))
            
            
            batch_predictions = (scores_ba - scores_ab).pow(2).sum(1).sqrt()
            print("batch dev prediction tensor", batch_predictions)
            print("batch dev label tensor", batch_dev_labels)
            batch_predictions = (batch_predictions > 4.45).detach().cpu()

            #scores_mean = (scores_ab + scores_ba) / 2

            #batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions), val_loss

def predict_cross_scores(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions_ab = []
    predictions_ba = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices).detach().cpu() 
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices).detach().cpu()

            #scores_mean = (scores_ab + scores_ba) / 2

            #batch_predictions = (scores_mean > 0.5).detach().cpu()
            batch_predictions_ab = scores_ab.detach().cpu()
            batch_predictions_ba = scores_ba.detach().cpu()
            predictions_ab.append(batch_predictions_ab)
            predictions_ba.append(batch_predictions_ba)
    return torch.cat(predictions_ab), torch.cat(predictions_ba)
    #return predictions

def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        output = torch.clamp(output,min=1e-8,max=1-1e-8)  
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))    


def train_wiki(features,
          dev_features,
          test_features,
          #dev_labels,
          parallel_model,
          #mention_map,
          working_folder,
          device,
          batch_size=32,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()
    #wiki_criterion = nn.CrossEntropyLoss(weight= torch.tensor([ 1,11.2])).to(device)
    #wiki_criterion = nn.CrossEntropyLoss().to(device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
#     optimizer = torch.optim.SGD([
#         {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class, 'momentum':0.9}
#     ])
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)

    new_batch_size = batch_size
     
    chunk_size = len(features)
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42)
    
    dev_indices = list(range(len(dev_features)))
    dev_pairs = list((dev_features[i] for i in dev_indices))
    dev_labels = torch.FloatTensor([x['label'] for x in dev_pairs]).to(device)
    
    input_ids = [dev_pairs [i]['input_ids'] for i in dev_indices]
    attention_mask = [dev_pairs[i]['attention_mask'] for i in dev_indices]
    position_ids = [dev_pairs[i]['position_ids'] for i in dev_indices]
    #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]

    input_ids = torch.stack(input_ids, dim =0).squeeze()
    attention_mask = torch.stack(attention_mask, dim =0).squeeze()
    position_ids = torch.stack(position_ids, dim =0).squeeze()
 
    tokenized_feature_dict_dev = {'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'position_ids': position_ids
                     }
    
    loss_values = []
    val_loss_values = []
    train_indices = list(range(len(features)))
    random.Random(42).shuffle(train_indices)
    for n in range(n_iters):

        random.Random(42).shuffle(train_indices)

        train_pairs = list((features[i] for i in train_indices))
        train_labels = torch.FloatTensor([x['label'] for x in train_pairs])
        iteration_loss = 0.
        for j in tqdm.tqdm(range(0, len(train_indices), chunk_size), desc='Creating batches for tokenizaton'):
          
            chunk_train_indices = train_indices[j: j + chunk_size]
            
            chunk_train_pairs= train_pairs[j: j + chunk_size]
            chunk_train_labels= train_labels[j: j + chunk_size]

            batch_indices = list(range(len(chunk_train_pairs)))
            
            
            input_ids = [train_pairs[i]['input_ids'] for i in chunk_train_indices]
            attention_mask = [train_pairs[i]['attention_mask'] for i in chunk_train_indices]
            position_ids = [train_pairs[i]['position_ids'] for i in chunk_train_indices]
      
             
            input_ids = torch.stack(input_ids, dim =0).squeeze()
            attention_mask = torch.stack(attention_mask, dim =0).squeeze()
            position_ids = torch.stack(position_ids, dim =0).squeeze()
      
            tokenized_feature_dict = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'position_ids': position_ids
                             }
       
            for i in tqdm.tqdm(range(0, len(chunk_train_indices), new_batch_size), desc='Training'):


                optimizer.zero_grad()

                mini_batch_indices = batch_indices[i: i + new_batch_size]
     
                
                scores, all_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices)
                token_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices, lm_only=True)
                
               
                print("scores", scores)
   
                batch_labels = torch.FloatTensor(chunk_train_labels[i: i + new_batch_size]) 
                batch_labels = batch_labels.type(torch.LongTensor)
                batch_labels = batch_labels.to(device)
                
                
                #scores_total = torch.squeeze(torch.sum(scores, dim=1))
                #scores_total = torch.FloatTensor(scores_total)
                #print("tota",scores_total.float())
                
                #loss = wiki_criterion(torch.squeeze(scores), batch_labels) 
                #loss = wiki_criterion(torch.squeeze(scores), batch_labels) 
                
         
                loss = 0.01*(bce_loss(torch.squeeze(scores), batch_labels)) +(cos_loss ( all_embed,token_embed, batch_labels))
         
                
                print("bce loss", bce_loss(torch.squeeze(scores), batch_labels))
                print("cosine loss", cos_loss ( all_embed,token_embed, batch_labels))
                print("training sample loss", loss)
                

                loss.backward()

                optimizer.step()

                iteration_loss += loss.item()


            print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
            loss_values.append(iteration_loss / len(train_pairs))
            
            # iteration accuracy

            dev_predictions, val_loss = predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size,dev_labels )

            dev_predictions = torch.squeeze(dev_predictions)
            print("dev predictions",dev_predictions )
            #val_loss_values.append(val_loss)
            print("Dev labels", dev_labels)
            #print("Dev predictions", dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
            
            
def train_qa_frozen(features,
          dev_features,
          test_features,
          #dev_labels,
          parallel_model,
          #mention_map,
          working_folder,
          device,
          force_lm_output=False,
                    
          batch_size=32,
          n_iters=10,
          
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
    optimizer = torch.optim.SGD([
       
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class, 'momentum':0.9}
    ])
#     optimizer = torch.optim.AdamW([
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
#     ])

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)


    new_batch_size = batch_size
     
    chunk_size = len(features)
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42)
    
    dev_indices = list(range(len(dev_features)))
    dev_pairs = list((dev_features[i] for i in dev_indices))
    dev_labels = torch.FloatTensor([x['label'] for x in dev_pairs])
    
    input_ids = [dev_pairs [i]['input_ids'] for i in dev_indices]
    attention_mask = [dev_pairs[i]['attention_mask'] for i in dev_indices]
    position_ids = [dev_pairs[i]['position_ids'] for i in dev_indices]
    #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]

    input_ids = torch.stack(input_ids, dim =0).squeeze()
    attention_mask = torch.stack(attention_mask, dim =0).squeeze()
    position_ids = torch.stack(position_ids, dim =0).squeeze()

    #print("input ids chunk shape ", input_ids.size())
    #print("attention mask chunk shape ", attention_mask.size())
    #print(" position_ids chunk shape ",  position_ids.size())

    tokenized_feature_dict_dev = {'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'position_ids': position_ids
                     }
    
    loss_values = []
    val_loss_values = []
    train_indices = list(range(len(features)))
    random.Random(42).shuffle(train_indices)
    for n in range(n_iters):
        
        #train_indices = list(range(len(features)))
       
        #random.shuffle(train_indices)
        random.Random(42).shuffle(train_indices)
        #print("initial train indices", train_indices)
#         train_pairs = list((train_pairs[i] for i in train_indices))
#         train_labels = list((train_labels[i] for i in train_indices))
        
        train_pairs = list((features[i] for i in train_indices))
        train_labels = torch.FloatTensor([x['label'] for x in train_pairs])
         
        

        iteration_loss = 0.
        for j in tqdm.tqdm(range(0, len(train_indices), chunk_size), desc='Creating batches for tokenizaton'):
            #print("Chunk size",chunk_size )
        
        
            chunk_train_indices = train_indices[j: j + chunk_size]
            
            chunk_train_pairs= train_pairs[j: j + chunk_size]
            chunk_train_labels= train_labels[j: j + chunk_size]
            
            #print(len(chunk_train_pairs))
            #print(len(chunk_train_labels))
            batch_indices = list(range(len(chunk_train_pairs)))
            
            
            input_ids = [train_pairs[i]['input_ids'] for i in chunk_train_indices]
            attention_mask = [train_pairs[i]['attention_mask'] for i in chunk_train_indices]
            position_ids = [train_pairs[i]['position_ids'] for i in chunk_train_indices]
            #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]
             
            input_ids = torch.stack(input_ids, dim =0).squeeze()
            attention_mask = torch.stack(attention_mask, dim =0).squeeze()
            position_ids = torch.stack(position_ids, dim =0).squeeze()
            
            #print("input ids chunk shape ", input_ids.size())
            #print("attention mask chunk shape ", attention_mask.size())
            #print(" position_ids chunk shape ",  position_ids.size())
            
            tokenized_feature_dict = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'position_ids': position_ids
                             }
            
            #generate the LM output here, 
            lm_out_dict_train = generate_lm_out_wiki(parallel_model, device, tokenized_feature_dict, batch_size)
            
            #print(lm_out_dict_train['ab'].size())
            #print("first batch indices", batch_indices)
            
            #train_ab, train_ba = tokenize(tokenizer, chunk_train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
#             dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
        
            #new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
            for i in tqdm.tqdm(range(0, len(chunk_train_indices), new_batch_size), desc='Training'):


                optimizer.zero_grad()

                #dev_pairs = dev_pairs[i: i + new_batch_size]



                

                #batch_indices = train_indices[i: i + new_batch_size]
                #batch_indices = batch_indices[i: i + new_batch_size]
                mini_batch_indices = batch_indices[i: i + new_batch_size]
                #print("next batch indices", mini_batch_indices)
                #once shuffled, now we do not need to original indices to create the mini batches 
                ab_out = lm_out_dict_train['ab'][mini_batch_indices, :]
                #print(ab_out.size())
                scores_ab  = parallel_model(ab_out.to(device), pre_lm_out=True)
                
                batch_labels = torch.FloatTensor(chunk_train_labels[i: i + new_batch_size]) 
                #batch_labels = batch_labels.type(torch.LongTensor)
                batch_labels = batch_labels.to(device)
                
                
                loss = bce_loss(torch.squeeze(scores_ab), batch_labels)
         

                loss.backward()

                optimizer.step()

                iteration_loss += loss.item()
                
 
            print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
            loss_values.append(iteration_loss / len(train_pairs))
            
            # iteration accuracy

            #dev_predictions, val_loss = predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size,dev_labels )
            
            dev_predictions, val_loss = frozen_predict(parallel_model, device, tokenized_feature_dict_dev ,
                                             batch_size, dev_labels, force_lm_output=True)
            dev_predictions = torch.squeeze(dev_predictions)
            val_loss_values.append(val_loss)
            print("Dev labels", dev_labels)
            print("Dev predictions", dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
    plt.plot(loss_values)
    plt.plot(val_loss_values) 
    return loss_values, val_loss_values
def train_qa(features,
          dev_features,
          test_features,
          #dev_labels,
          parallel_model,
          #mention_map,
          working_folder,
          device,
          batch_size=32,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
#     optimizer = torch.optim.SGD([
#         {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class, 'momentum':0.9}
#     ])
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)
  

    new_batch_size = batch_size
     
    chunk_size = len(features)
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42)
    
    dev_indices = list(range(len(dev_features)))
    dev_pairs = list((dev_features[i] for i in dev_indices))
    dev_labels = torch.FloatTensor([x['label'] for x in dev_pairs])
    
    input_ids = [dev_pairs [i]['input_ids'] for i in dev_indices]
    attention_mask = [dev_pairs[i]['attention_mask'] for i in dev_indices]
    position_ids = [dev_pairs[i]['position_ids'] for i in dev_indices]
    #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]

    input_ids = torch.stack(input_ids, dim =0).squeeze()
    attention_mask = torch.stack(attention_mask, dim =0).squeeze()
    position_ids = torch.stack(position_ids, dim =0).squeeze()

    #print("input ids chunk shape ", input_ids.size())
    #print("attention mask chunk shape ", attention_mask.size())
    #print(" position_ids chunk shape ",  position_ids.size())

    tokenized_feature_dict_dev = {'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'position_ids': position_ids
                     }
    
    loss_values = []
    val_loss_values = []
    train_indices = list(range(len(features)))
    random.Random(42).shuffle(train_indices)
    for n in range(n_iters):
        
        #train_indices = list(range(len(features)))
       
        #random.shuffle(train_indices)
        random.Random(42).shuffle(train_indices)
        #print("initial train indices", train_indices)
#         train_pairs = list((train_pairs[i] for i in train_indices))
#         train_labels = list((train_labels[i] for i in train_indices))
        
        train_pairs = list((features[i] for i in train_indices))
        train_labels = torch.FloatTensor([x['label'] for x in train_pairs])
         
        

        iteration_loss = 0.
        for j in tqdm.tqdm(range(0, len(train_indices), chunk_size), desc='Creating batches for tokenizaton'):
            #print("Chunk size",chunk_size )
        
        
            chunk_train_indices = train_indices[j: j + chunk_size]
            
            chunk_train_pairs= train_pairs[j: j + chunk_size]
            chunk_train_labels= train_labels[j: j + chunk_size]
            
            #print(len(chunk_train_pairs))
            #print(len(chunk_train_labels))
            batch_indices = list(range(len(chunk_train_pairs)))
            
            
            input_ids = [train_pairs[i]['input_ids'] for i in chunk_train_indices]
            attention_mask = [train_pairs[i]['attention_mask'] for i in chunk_train_indices]
            position_ids = [train_pairs[i]['position_ids'] for i in chunk_train_indices]
            #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]
             
            input_ids = torch.stack(input_ids, dim =0).squeeze()
            attention_mask = torch.stack(attention_mask, dim =0).squeeze()
            position_ids = torch.stack(position_ids, dim =0).squeeze()
            
            #print("input ids chunk shape ", input_ids.size())
            #print("attention mask chunk shape ", attention_mask.size())
            #print(" position_ids chunk shape ",  position_ids.size())
            
            tokenized_feature_dict = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'position_ids': position_ids
                             }
            

            #print("first batch indices", batch_indices)
            
            #train_ab, train_ba = tokenize(tokenizer, chunk_train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
#             dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
        
            #new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
            for i in tqdm.tqdm(range(0, len(chunk_train_indices), new_batch_size), desc='Training'):


                optimizer.zero_grad()

                #dev_pairs = dev_pairs[i: i + new_batch_size]



                

                #batch_indices = train_indices[i: i + new_batch_size]
                #batch_indices = batch_indices[i: i + new_batch_size]
                mini_batch_indices = batch_indices[i: i + new_batch_size]
                #print("next batch indices", mini_batch_indices)
                #once shuffled, now we do not need to original indices to create the mini batches 
                
                
                #print("batch indices", batch_indices)

                #train_pairs = list((train_pairs[i] for i in batch_indices))
                #print(train_pairs)
                #train_pairs = train_pairs[batch_indices]

                
                #print(train_ab)
                
                scores, all_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices)
                token_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices, lm_only=True)
                #print("scores", scores)
                #print("logits", logits)
                
                #scores_ba = forward_ab(parallel_model, train_ba, device, mini_batch_indices)

                #batch_labels = train_labels[batch_indices].to(device)
      
                #loss = bce_loss(torch.squeeze(scores_mean), batch_labels) + mse_loss(scores_ab, scores_ba)
                #loss = bce_loss(torch.squeeze(scores), batch_labels) + cos_loss ( all_embed,token_embed, batch_labels)
                loss = 0.01*(bce_loss(torch.squeeze(scores), batch_labels)) +(cos_loss ( all_embed,token_embed, batch_labels))
                #loss = cos_loss ( all_embed,token_embed, batch_labels)
                
                print("bce loss", bce_loss(torch.squeeze(scores), batch_labels))
                print("cosine loss", cos_loss ( all_embed,token_embed, batch_labels))
                print("training sample loss", loss)
                

                loss.backward()

                optimizer.step()

                iteration_loss += loss.item()
                



            print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
            loss_values.append(iteration_loss / len(train_pairs))
            
            # iteration accuracy

            dev_predictions, val_loss = predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size,dev_labels )
            dev_predictions = torch.squeeze(dev_predictions)
            val_loss_values.append(val_loss)
            print("Dev labels", dev_labels)
            print("Dev predictions", dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
    plt.plot(loss_values)
    plt.plot(val_loss_values)           
            
if __name__ == '__main__':
    
    as_sectitle_dir = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data/wiki-section-titles/'

    as_wiki = WikiSectionTitles(as_sectitle_dir)
    as_wiki_train =  as_wiki.get_train_examples('as')
    as_wiki_dev = as_wiki.get_dev_examples('as')
    as_wiki_test = as_wiki.get_test_examples('as')

    len(as_wiki_train), len(as_wiki_dev), len(as_wiki_test) #check the sizes of the three splits 

    # create a list of labels before converting into features 
    label_list = []
    for i, j in enumerate(as_wiki_train):
        label_list.append(j.label)


    label_list = set(label_list)
    label_list = sorted(label_list)


    train_features, train_label_map = convert_multiple_choice_examples_to_features(
        list(as_wiki_train),
        tokenizer,
        512,
        list(label_list),
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) 

    dev_features, dev_label_map = convert_multiple_choice_examples_to_features(
        list(as_wiki_dev),
        tokenizer,
        512,
        list(label_list),
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) 


    test_features,test_label_map = convert_multiple_choice_examples_to_features(
        list(as_wiki_test),
        tokenizer,
        512,
        list(label_list),
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) 
    
    parent_path = '/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/'
    working_folder = parent_path + "/parsing/ecb"
    scorer_folder = working_folder + '/scorer/chk_9/'
    model_name = scorer_folder + 'bert'
    linear_weights_path = scorer_folder + 'linear.chkpt'
    

 
    
   
    
    
    device = torch.device('cuda:0')
    #model_name = 'allenai/longformer-base-4096'
    model_name = "/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500"
    #model_name = 'biu-nlp/cdlm'
    #linear_weights = torch.load(linear_weights_path)
    #scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name, linear_weights=linear_weights).to(device)
    #scorer_module = LongFormerCrossEncoder(is_training=True, model_name=model_name).to(device)
    #scorer_module = LongFormerTriplet(is_training=True, model_name=model_name).to(device)
    scorer_module =AxBERTa_Classfier(is_training=True, model_name=model_name).to(device)
    
    
    
   

#     #device_ids = list(range(2))
    device_ids = [0]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    #print(parallel_model)
    working_folder = parent_path + "/parsing/ecb"

#     # edit this or not!
#     ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"

     # create the features as a binary classification problem 
    
    parallel_model.module.tokenizer.model_max_length = 514
    tokenizer = parallel_model.module.tokenizer
    #train_features = pickle.load(open("train_wiki_features_upsampled", 'rb'))
    #split datasets into train, test and validation sets. 
    

    
    
# # 3 * 10e(-5)
    train_qa(train_features,
          dev_features,
          test_features,
          #dev_labels,
          parallel_model,
          #mention_map,
          working_folder,
          device,
          batch_size=60,
          n_iters=10,
          lr_lm=0.00002,
          lr_class=0.00001)
 





 


