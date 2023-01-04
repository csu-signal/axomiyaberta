
 

import os
import sys
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
from qa_models import AxBERTa_EmbeddingDisperser
 
import random
from tqdm.autonotebook import tqdm
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
 
wiki_cloze_dir = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data/wiki-cloze'



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




class WikiCloze(DataProcessor):
    """Processor for Wiki Cloze QA dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def modes(self):
        return ['test']

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath)['cloze_data'], 'test')

    def get_labels(self, lang):
        """See base class."""
        return list(range(4))

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, item) in enumerate(items):
            if '' in [option.strip() for option in item['options']]:
                continue
            example = MultipleChoiceExample(
                example_id=i,
                #question=item['question'].replace('<MASK>', '[MASK]'),
                question=item['question'].replace('<MASK>', '<m>[MASK]</m>'),
                contexts=[],
                endings=item['options'],
                label=item['options'].index(item['answer'])
            )
            examples.append(example)
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
    pan_features=None,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    count = 0
    features = defaultdict(dict)
    sentences = []
    
    
   
    
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
     
        batch_label = [example.label]*4
        for ending_idx, (ending, label) in enumerate(zip(example.endings, batch_label)):

            text_b = example.question + " " + "<m>" + ending + "</m>" # add the special tokens out here 
     
            inputs = tokenizer(
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation='longest_first',
                pad_to_max_length=True,return_tensors = 'pt'
            )
       
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )
            features[count]["input_ids"] = inputs["input_ids"]
            features[count]["attention_mask"] = inputs["attention_mask"]
            features[count]["token_type_ids"] = inputs["token_type_ids"]
            #features[count]["positions_ids"] = torch.arange(len(inputs['input_ids'])) 
            features[count]["position_ids"] = torch.arange(inputs["input_ids"].shape[-1]).expand(inputs["input_ids"].shape)
            #.expand(inputs["input_ids"].shape)
            
            
            
            features[count]["label"] = 1 if ending_idx == label else 0
            features[count]["pan_features"] = pan_features[count]
            count+=1
            
  
    return features  
 
def convert_int(x):
    
    a = x.replace("[","").replace("]","").split(",")
    
    desired_array = [int(x) for x in a]
    #a = np.array([int(x) for x in a])
    return desired_array 


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


def load_data(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            triviality_label = 0 if row[2] =='HARD' else 1
            
            all_examples.append((mention_pair, triviality_label))

    return all_examples

#load lemma balanced TP and FP tsv pairs

def load_data_tp_fp(trivial_non_trivial_path):
    all_examples = []
    pos = []
    neg = []
    with open(trivial_non_trivial_path) as tnf:
        rd = csv.reader(tnf, delimiter="\t", quotechar='"')
        
        for line in rd:
            #row = line.strip().split(',')
            mention_pair = line[:2]
            #print(line[2])
            #print(mention_pair)
            if line[2] =='POS':
                triviality_label = 1
                all_examples.append((mention_pair, triviality_label))
                #pos.append(mention_pair)
                
            else:
                triviality_label = 0
                all_examples.append((mention_pair, triviality_label))
                #neg.append(mention_pair)
          
    return all_examples 

def load_data_cross_full(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            triviality_label = 0 if row[3] =='NEG' else 1

            all_examples.append((mention_pair, triviality_label))

    return all_examples

def load_data_pair_coref_dev(trivial_non_trivial_path):
    all_examples = []
    #condition to select only hard pos and hard neg examples 
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            if row[2]=='HARD':
                #triviality_label = int(row[3])
                triviality_label = 0 if row[3] =='NEG' else 1

                all_examples.append((mention_pair, triviality_label))

    return all_examples
def load_data_pair_coref(trivial_non_trivial_path):
    all_examples = []
    #condition to select only hard pos and hard neg examples 
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            if row[2]=='HARD':
                
            #triviality_label = int(row[3])
                triviality_label = -1 if row[3] =='NEG' else 1

                all_examples.append((mention_pair, triviality_label))

    return all_examples


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


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False, panphon_features=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    batch_panphon_features = ab_dict['panphon_features'][indices, :]
    
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)
    batch_panphon_features.to(device)
    if panphon_features:
        return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only, panphon_features = batch_panphon_features)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only, panphon_features = None)

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
def generate_lm_out(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    ab_lm_out_all = []
    ba_lm_out_all = []
    new_batch_size = batching(n, batch_size, len(device_ids))
    batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Generating LM Outputs"):
            batch_indices = indices[i: i + batch_size]
            lm_out_ab = forward_ab(parallel_model, dev_ab, device, batch_indices, lm_only=True).detach().cpu()
            ab_lm_out_all.append(lm_out_ab)

            lm_out_ba = forward_ab(parallel_model, dev_ba, device, batch_indices, lm_only=True).detach().cpu()
            ba_lm_out_all.append(lm_out_ba)

    return {'ab': torch.vstack(ab_lm_out_all), 'ba': torch.vstack(ba_lm_out_all)}



def predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size, dev_labels, test=False):
    
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()

    n = tokenized_feature_dict_dev['input_ids'].shape[0]
    weights = torch.FloatTensor([1.0, 3.2])
    
    
#     dev_labels = dev_labels.to(device)
    indices = list(range(n))
    predictions = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    parallel_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            batch_dev_labels = dev_labels[i: i + batch_size].to(device)

            scores_ab = forward_ab(parallel_model, tokenized_feature_dict_dev, device, batch_indices, lm_only=True,panphon_features=False ) 
            scores, scores_ba = forward_ab(parallel_model, tokenized_feature_dict_dev, device, batch_indices, panphon_features=False)
            
            #val_loss = 0.01*(bce_loss(torch.squeeze(scores), batch_dev_labels)) +(cos_loss ( scores_ba,scores_ab, batch_dev_labels))
            val_loss = 0.01*( custom_bceloss(torch.squeeze(scores), batch_dev_labels,weights )) +(cos_loss ( scores_ba,scores_ab, batch_dev_labels))
            
           
            
            
            batch_predictions = (scores_ba - scores_ab).pow(2).sum(1).sqrt()
            print("batch dev prediction tensor", batch_predictions) if test is False else print("batch test prediction tensor", batch_predictions)
            print("batch dev label tensor", batch_dev_labels) if test is False else  print("batch test label tensor", batch_dev_labels)
            batch_predictions = (batch_predictions > 4.45).detach().cpu()

            #scores_mean = (scores_ab + scores_ba) / 2

            #batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions), val_loss
 
def custom_bceloss(input, target, weights):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

def train(features,
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
    weights = torch.FloatTensor([1.0, 3.2])
    
    
   

   

          
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
#     optimizer = torch.optim.SGD([
#         {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class, 'momentum':0.9}
#     ])
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])
 
    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)
 

    new_batch_size = batch_size
     
    chunk_size = len(features)
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42) # might be unnecessary here!
    
    dev_indices = list(range(len(dev_features)))
    dev_pairs = list((dev_features[i] for i in dev_indices))
    dev_labels = torch.FloatTensor([x['label'] for x in dev_pairs])
    
    input_ids = [dev_pairs [i]['input_ids'] for i in dev_indices]
    attention_mask = [dev_pairs[i]['attention_mask'] for i in dev_indices]
    position_ids = [dev_pairs[i]['position_ids'] for i in dev_indices]
    panphon_features = torch.tensor([dev_pairs[i]['pan_features'] for i in dev_indices],  dtype=torch.long)
    
    
   
    
 
    #idx = [train_pairs[i]['label'] for i in train_indices[0:5]]

    input_ids = torch.stack(input_ids, dim =0).squeeze()
    attention_mask = torch.stack(attention_mask, dim =0).squeeze()
    position_ids = torch.stack(position_ids, dim =0).squeeze()
    #panphon_features = torch.stack(panphon_features, dim =0).squeeze()

    print("input ids chunk shape ", input_ids.size())
    print("attention mask chunk shape ", attention_mask.size())
    print(" position_ids chunk shape ",  position_ids.size())
    print("panphon features chunk shape ",  panphon_features.size())

    tokenized_feature_dict_dev = {'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'position_ids': position_ids,
                     'panphon_features': panphon_features,
                                  
                     }
    
    test_indices = list(range(len(test_features)))
    test_pairs = list((test_features[i] for i in test_indices))
    test_labels = torch.FloatTensor([x['label'] for x in test_pairs])
    
    input_ids = [test_pairs [i]['input_ids'] for i in test_indices]
    attention_mask = [test_pairs[i]['attention_mask'] for i in test_indices]
    position_ids = [test_pairs[i]['position_ids'] for i in test_indices]
    panphon_features = torch.tensor([test_pairs[i]['pan_features'] for i in test_indices],  dtype=torch.long)
    
    input_ids = torch.stack(input_ids, dim =0).squeeze()
    attention_mask = torch.stack(attention_mask, dim =0).squeeze()
    position_ids = torch.stack(position_ids, dim =0).squeeze()
    
    
    print("input ids TEST chunk shape ", input_ids.size())
    print("attention TEST mask chunk shape ", attention_mask.size())
    print(" position_ids TEST chunk shape ",  position_ids.size())
    print("panphon features TEST chunk shape ",  panphon_features.size())

    tokenized_feature_dict_test = {'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'position_ids': position_ids,
                     'panphon_features': panphon_features,
                                  
                     }
    
    loss_values = []
    val_loss_values = []
    test_loss_values = []
    train_indices = list(range(len(features)))
    
    random.Random(42).shuffle(train_indices)
    for n in range(n_iters):
        
        #random.shuffle(train_indices)
        random.Random(42).shuffle(train_indices)

        train_pairs = list((features[i] for i in train_indices))
        train_labels = torch.FloatTensor([x['label'] for x in train_pairs])
         
        

        iteration_loss = 0.
        for j in tqdm(range(0, len(train_indices), chunk_size), desc='Creating batches for tokenizaton'):
 
            chunk_train_indices = train_indices[j: j + chunk_size]
            
            chunk_train_pairs= train_pairs[j: j + chunk_size]
            chunk_train_labels= train_labels[j: j + chunk_size]

            batch_indices = list(range(len(chunk_train_pairs)))
            
            
            input_ids = [train_pairs[i]['input_ids'] for i in chunk_train_indices]
            attention_mask = [train_pairs[i]['attention_mask'] for i in chunk_train_indices]
            position_ids = [train_pairs[i]['position_ids'] for i in chunk_train_indices]
            train_panphon_features = torch.tensor([train_pairs[i]['pan_features'] for i in chunk_train_indices],  dtype=torch.long)
 
           
            print("pan trainfeaturs size",train_panphon_features.size())
            
             
            input_ids = torch.stack(input_ids, dim =0).squeeze()
            attention_mask = torch.stack(attention_mask, dim =0).squeeze()
            position_ids = torch.stack(position_ids, dim =0).squeeze()
            
 
            tokenized_feature_dict = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'position_ids': position_ids,
                                      'panphon_features': train_panphon_features,
                                      
                            
                             }
            

            for i in tqdm(range(0, len(chunk_train_indices), new_batch_size), desc='Training'):


                optimizer.zero_grad()

                mini_batch_indices = batch_indices[i: i + new_batch_size]

                
                scores, all_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices,panphon_features=False )
                token_embed = forward_ab(parallel_model, tokenized_feature_dict, device, mini_batch_indices, lm_only=True,panphon_features=False)
   
                batch_labels = torch.FloatTensor(chunk_train_labels[i: i + new_batch_size]).to(device)
             
                #loss = 0.01*(bce_loss(torch.squeeze(scores), batch_labels)) +(cos_loss ( all_embed,token_embed, batch_labels))
                loss = 0.01*(custom_bceloss(torch.squeeze(scores), batch_labels,weights )) +(cos_loss ( all_embed,token_embed, batch_labels))
            
            
                
                #loss = cos_loss ( all_embed,token_embed, batch_labels)
                
                print("bce loss", custom_bceloss(torch.squeeze(scores), batch_labels,weights ))
                print("cosine loss", cos_loss ( all_embed,token_embed, batch_labels))
                print("training sample loss", loss)
                

                loss.backward()

                optimizer.step()

                iteration_loss += loss.item()

            print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
            loss_values.append(iteration_loss / len(train_pairs))
            
            # iteration accuracy

            dev_predictions, val_loss = predict_qa(parallel_model, device, tokenized_feature_dict_dev, batch_size,dev_labels , test=False)
            dev_predictions = torch.squeeze(dev_predictions)
            val_loss_values.append(val_loss)
            print("Dev labels", dev_labels)
            print("Dev predictions", dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
            
            
            test_predictions, test_loss = predict_qa(parallel_model, device, tokenized_feature_dict_test, batch_size,test_labels, test=True )
            test_predictions = torch.squeeze(test_predictions)
            test_loss_values.append(test_loss)
            
            
            
            print("Test labels", test_labels)
            print("Test predictions", test_predictions)

            print("Test accuracy:", accuracy(test_predictions, test_labels))
            print("Test precision:", precision(test_predictions, test_labels))
            print("Test f1:", f1_score(test_predictions, test_labels))
            
            
            
            
            
            scorer_folder = working_folder + f'/scorer_asqa/chk_{n}'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)
            model_path_linear = scorer_folder + '/linear.chkpt'
            model_path_embed = scorer_folder + '/embed.chkpt'
            model_path_option = scorer_folder + '/option.chkpt'
            
            
            
            torch.save(parallel_model.module.linear.state_dict(), model_path_linear)
            
            torch.save(parallel_model.module.embed .state_dict(), model_path_embed)
            torch.save(parallel_model.module.option.state_dict(),  model_path_option)
            parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
            parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

#         scorer_folder = working_folder + '/scorer_asqa/'
#         if not os.path.exists(scorer_folder):
#             os.makedirs(scorer_folder)
#         model_path = scorer_folder + '/linear.chkpt'
#         torch.save(parallel_model.module.linear.state_dict(), model_path)
#         torch.save(parallel_model.module.linear.state_dict(), model_path)
#         torch.save(parallel_model.module.linear.state_dict(), model_path)
        
        
#         parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#         parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
 
    plt.plot(loss_values)
    plt.plot(val_loss_values)
    plt.plot(test_loss_values)
    
#     with open("train_loss", "wb") as fp:
#         pickle.dump(loss_values, fp)
        
#     with open("val_loss", "wb") as fp:
#         pickle.dump(val_loss_values, fp)
        

#     plt.savefig('train_loss.png')


#         scorer_folder = working_folder + f'/scorer_asqa/chk_{n}'
#         if not os.path.exists(scorer_folder):
#             os.makedirs(scorer_folder)
#         #model_path = scorer_folder + '/linear.chkpt'
#         #torch.save(parallel_model.module.linear.state_dict(), model_path)
#         parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#         parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

#     scorer_folder = working_folder + '/scorer_asqa/'
#     if not os.path.exists(scorer_folder):
#         os.makedirs(scorer_folder)
#     #model_path = scorer_folder + '/linear.chkpt'
#     #torch.save(parallel_model.module.linear.state_dict(), model_path)
#     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
 
def batching(n, batch_size, min_batch):
    new_batch_size = batch_size
    while n % new_batch_size < min_batch:
        new_batch_size -= 1
    return new_batch_size




if __name__ == '__main__':
    

    
    
    cloze = WikiCloze(wiki_cloze_dir)
    examples = cloze.get_test_examples('as')
    labels = []
    for i, j in enumerate(examples):
        labels.append(j.label)
    
   
    
    
    device = torch.device('cuda:0')
    #model_name = 'allenai/longformer-base-4096'
    model_name = "/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500"

    #scorer_module =AxBERTa_EmbeddingDisperser(is_training=True,  pan=True,pan_features=None,max_pad_len=360, model_name=model_name).to(device)
    scorer_module =AxBERTa_EmbeddingDisperser(is_training=True,  pan=False,pan_features=None,max_pad_len=360, model_name=model_name).to(device)
  
    
    working_folder = '/s/chopin/d/proj/ramfis-aida/axbert/axomiyaberta/axomiyaberta/task_finetune/QA_multiplechoice/'  
    
   

#     #device_ids = list(range(2))
    device_ids = [0]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    parallel_model.module.tokenizer.model_max_length = 514
    tokenizer = parallel_model.module.tokenizer
    
    #split datasets into train, test and validation sets. 
    
    train_examples = examples[0:2000]
    print(len(train_examples))
    dev_examples = examples[2000:2500]
    print(len(dev_examples))
    test_examples = examples[2500:len(examples)]
    
    train_labels = labels[0:2000]
    dev_labels = labels[2000:2500]
    test_labels = labels[2500:len(examples)]
    
    
    #load the panphon features accordingly 
    
    
    import pandas as pd
    as_cloze_pan = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data/wiki-cloze/qa panphon_features.csv'

    as_cloze_pan = pd.read_csv(as_cloze_pan)


    len(as_cloze_pan) 
    #load and trim panphone features
    as_cloze_pan["new_label"] = as_cloze_pan['Panphon features'].apply(lambda x: convert_int(x) )
    
    #make train, dev and test for panphon features of candidate options. 
    
    as_cloze_pan_train = list(as_cloze_pan["new_label"] )[0:8000]
    as_cloze_pan_dev = list(as_cloze_pan["new_label"] )[8000:10000]
    as_cloze_pan_test = list(as_cloze_pan["new_label"] )[10000:len(as_cloze_pan)]
    
    
    train_features= convert_multiple_choice_examples_to_features(
    train_examples ,
    tokenizer,
    512,
    train_labels,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    pan_features = as_cloze_pan_train,
)
    
    print("train features created", len(train_features))
    dev_features= convert_multiple_choice_examples_to_features(
    dev_examples ,
    tokenizer,
    512,
    dev_labels,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    pan_features = as_cloze_pan_dev,
)
    print("dev features created", len(dev_features))
    
    test_features= convert_multiple_choice_examples_to_features(
    test_examples,
    tokenizer,
    512,
    test_labels,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    pan_features = as_cloze_pan_test,
)
    
    print("test features created", len(test_features))
    

    train(train_features,
          dev_features,
          test_features,
          #dev_labels,
          parallel_model,
          #mention_map,
          working_folder,
          device,
          batch_size=80,
          n_iters=20,
          lr_lm=0.00002,
          lr_class=0.0001)
    






 