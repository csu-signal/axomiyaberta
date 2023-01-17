#Author: Abhijnan Nath. Extended from https://github.com/AI4Bharat/indic-bert and  huggingface: https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

import os
import sys
import gc
import numpy as np
import pandas as pd
 # for reproducibility 
import csv
gc.collect()
import torch
import torch.nn as nn
import torch.nn as F
from collections import defaultdict

from evaluate_metrics_ner import * # import all the custom evaluation metrics for ANSER dataset, works for both ASNER and WIKI NER format
from datasets import load_dataset, load_metric
import numpy as np
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer

 
from transformers import *
from transformers import AutoModel, AutoTokenizer, AutoModelForMultipleChoice, AutoModelForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.optim import AdamW
from transformers import get_scheduler
 
import numpy as np
import pyhocon
import os
from inspect import getfullargspec


from transformers import (AutoModelWithLMHead, 
                          AutoTokenizer, 
                          BertConfig)
import argparse
import logging
import os
import glob
import random
import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
#from tqdm.autonotebook import tqdm
#from tqdm.auto import tqdm
from tqdm import tqdm
import csv

#from ..data import load_dataset
#from ..data.examples import *
logger = logging.getLogger(__name__)
model_name="/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500"
import csv
import json
import os
import tqdm
import logging

from dataclasses import dataclass
from typing import Optional, List, Any, Union
from transformers import PreTrainedTokenizer
logger = logging.getLogger(__name__)
from collections import defaultdict

# from examples import MultipleChoiceExample, TextExample, TokensExample

parent_path = '/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/'
parent_path_data = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data'
wiki_cloze_dir = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data/wiki-cloze'
wiki_ner_dir = '/s/chopin/d/proj/ramfis-aida/axbert/As_Indic_data/wikiann-ner'



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


# class IndicNLPHeadlines(DataProcessor):
#     """Processor for the Headline Predction dataset"""

#     def __init__(self, data_dir):
#         self.data_dir = data_dir

#     def get_train_examples(self, lang):
#         """See base class."""
#         fname = '{}/{}-train.json'.format(lang, lang)
#         fpath = os.path.join(self.data_dir, fname)
#         return self._create_examples(self.read_json(fpath), 'train')

#     def get_dev_examples(self, lang):
#         '''See base class.'''
#         fname = '{}/{}-valid.json'.format(lang, lang)
#         fpath = os.path.join(self.data_dir, fname)
#         return self._create_examples(self.read_json(fpath), 'dev')

#     def get_test_examples(self, lang):
#         '''See base class.'''
#         fname = '{}/{}-test.json'.format(lang, lang)
#         fpath = os.path.join(self.data_dir, fname)
#         return self._create_examples(self.read_json(fpath), 'test')

#     def get_labels(self, lang):
#         """See base class."""
#         return ['A', 'B', 'C', 'D']

#     def _create_examples(self, items, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = [
#             MultipleChoiceExample(
#                 example_id=idx,
#                 question='',
#                 contexts=[item['content'], item['content'], item['content'],
#                           item['content']],
#                 endings=[item['optionA'], item['optionB'], item['optionC'],
#                          item['optionD']],
#                 label=item['correctOption'],
#             )
#             for idx, item in enumerate(items)
#         ]
#         return examples


# class WikiCloze(DataProcessor):
#     """Processor for Wiki Cloze QA dataset"""

#     def __init__(self, data_dir):
#         self.data_dir = data_dir

#     def modes(self):
#         return ['test']

#     def get_test_examples(self, lang):
#         """See base class."""
#         fname = '{}.json'.format(lang, lang)
#         fpath = os.path.join(self.data_dir, fname)
#         return self._create_examples(self.read_json(fpath)['cloze_data'], 'test')

#     def get_labels(self, lang):
#         """See base class."""
#         return list(range(4))

#     def _create_examples(self, items, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, item) in enumerate(items):
#             if '' in [option.strip() for option in item['options']]:
#                 continue
#             example = MultipleChoiceExample(
#                 example_id=i,
#                 #question=item['question'].replace('<MASK>', '[MASK]'),
#                 question=item['question'].replace('<MASK>', '<m> [MASK] </m>'),
#                 contexts=[],
#                 endings=item['options'],
#                 label=item['options'].index(item['answer'])
#             )
#             examples.append(example)
#         return examples


class WikiNER(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, lang, mode):
        #mode = 'valid' if mode == 'dev' else mode
        mode = 'valid' if mode == 'dev' else mode
        file_path = os.path.join(self.data_dir, lang, f'{mode}.txt')
        #file_path = os.path.join(self.data_dir, f'{mode}.txt')
        guid_index = 1
        examples = []
        with open(file_path, encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if words:
                        example = TokensExample(
                            guid=f'{mode}-{guid_index}',
                            words=words,
                            labels=labels
                        )
                        examples.append(example)
                    guid_index += 1
                    words = []
                    labels = []
                else:
                    #print("yes this split")
                    splits = line.split(' ')
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace('\n', ''))
                    else:
                        # Examples could have no label for mode = 'test'
                        labels.append('O')
            if words:
                example = TokensExample(
                    guid=f'{mode}-{guid_index}',
                    words=words,
                    labels=labels
                )
                examples.append(example)
        return examples

    def get_labels(self, lang):
        path = os.path.join(self.data_dir, lang, 'labels.txt')
        with open(path, 'r') as f:
            labels = f.read().splitlines()
        if 'O' not in labels:
            labels = ['O'] + labels
        return labels

class ASNER(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, lang, mode):
        #mode = 'valid' if mode == 'dev' else mode
        mode = 'dev' if mode == 'valid' else mode
        file_path = os.path.join(self.data_dir, f'{mode}.txt')
        #file_path = os.path.join(self.data_dir, f'{mode}.txt')
        guid_index = 1
        count = 0
        examples = []
        with open(file_path, encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    #print("line starts")
                    if words:
                        #print("wordstart")
                        example = TokensExample(
                            guid=f'{mode}-{guid_index}',
                            words=words,
                            labels=labels
                        )
                        examples.append(example)
                    guid_index += 1
                    words = []
                    labels = []
                else:
                    count+=1
                    #print("yes this split",count )
                
                    #splits = line.split(' ')
                    splits = line.split("\t")
                    #plits = line.split("\t")
                    words.append(splits[0])
                    if len(splits) ==2 and line !='\n':
                        bad_label = splits[-1].replace('\n', '')
#                         if bad_label =='':
# #                             print("unindentified label")
# #                             print("line",line)
#                             print(len(splits))
                            
                            
                        labels.append(splits[-1].replace('\n', ''))
                        #print(len(splits))
                        
                        #labels.append(splits[-1])
                    elif len(splits) ==3:
                         
                        #print("goodlabels", splits[1].replace('\n', ''))
                        labels.append(splits[-1].replace('\n', ''))
                    
                    else:
                        count+=1
                        #print("O label ", count)
                        
                        # Examples could have no label for mode = 'test'
                        labels.append('O')
            if words:
                example = TokensExample(
                    guid=f'{mode}-{guid_index}',
                    words=words,
                    labels=labels
                )
                examples.append(example)
        return examples
    
    
    def cleanify_tokens(self, examples):
        
        c=0
        token = []
        for i, x in enumerate(examples):
            for j, (z,y) in enumerate(zip(x.words,x.labels)):
                if y=='' and z=='':
                  

                    trim_tok = x.words 
                    trim_lab = x.labels 
                    trim_tok.remove('')
                    trim_lab.remove('')

                    examples[i].labels = trim_lab
                    examples[i].words =  trim_tok
                 
                    c+=1

                    token.append((z,y))

              
                elif y=='':
                    if j==0:
                        new_label = examples[i].labels[j+1]
                        examples[i].labels[j] = new_label
                    elif j>0:
                        new_label = examples[i].labels[j-1]
                        examples[i].labels[j] = new_label
                        c+=1
                        token.append((z,y))




        return examples 
        
      

# tensor dataset loader

@dataclass
class NER_InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: None
    attention_mask: None
    token_type_ids: Any = None
    label: Any = None

@dataclass

 
class TokensExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This
        should be specified for train and dev examples, but not for test
        examples.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]
        
        
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

def flatten(l):
    result = []
    for element in l:
        if hasattr(element, "__iter__") and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result



def convert_tokens_examples_to_features(
    examples: List[TokensExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token='[CLS]',
    cls_token_segment_id=0,
    sep_token='[SEP]',
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    full_token =False,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}
    print("labelmap", label_map)

    features = []
    features_dict = defaultdict(dict)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for i , (word, label) in enumerate(zip(example.words, example.labels)):
  
            word_tokens = tokenizer.tokenize(word)
        


            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                #print("yes")
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                #print(ex_index)
                
                
                if full_token == True:
                    label_ids.extend([[label_map[label]]*len(word_tokens) ])
                    label_ids  = flatten(label_ids)
                else:
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    
                   
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            print("max legnth yes")
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        #print("initital tokens", tokens)
        tokens.append(tokenizer.sep_token)


        label_ids += [pad_token_label_id]
     
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens  
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
   
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
 

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
          

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
   


        if "token_type_ids" not in tokenizer.model_input_names:
            #print("true")
            segment_ids = None
        features_dict[ex_index]["input_ids"] = torch.tensor(input_ids).unsqueeze(-1)
        features_dict[ex_index]["attention_mask"] = torch.tensor(input_mask).unsqueeze(-1)
        features_dict[ex_index]["segment_ids"] = torch.tensor(segment_ids).unsqueeze(-1)
        #features[ex_index]["positions_ids"] = torch.arange(len(input_ids) )
        #features_dict[ex_index]["position_ids"] = torch.arange(input_ids.shape[-1]).expand(input_ids.shape)
        features_dict[ex_index]["label_ids"] =torch.tensor(label_ids).unsqueeze(-1)

        features.append(
            NER_InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label=label_ids
            )
        )
        
    return features, features_dict, label_map


def make_loader(features, batch_size, shuffle=True):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids or 0 for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # all_candidates = torch.tensor([f.candidates for f in features], dtype=torch.long)
   
    
#     elif self.hparams['output_mode'] == 'regression':
#         all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    return DataLoader(
        TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
        # TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_candidates),
        batch_size=batch_size,shuffle=shuffle,
    )

def train_dataloader(self):
    train_batch_size = self.hparams['train_batch_size']
    train_features = self.load_features('train')
    dataloader = self.make_loader(train_features, train_batch_size)

    t_total = (
        (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams['n_gpu'])))
            // self.hparams['gradient_accumulation_steps']
            * float(self.hparams['num_train_epochs'])
        )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams['warmup_steps'], num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader
def compute_metrics(predictions, labels, id2label):
 
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

	# Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
 
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
 
    scores = precision_recall_fscore_support_asner(true_labels, true_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels,true_predictions)
    
    return scores,  accuracy
 
metric = load_metric("seqeval")

def postprocess(predictions, labels, label_names):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions



if __name__ == '__main__':
    
    
    #load datasets
    asner_dir = '../../data/'
    
    asner = ASNER(asner_dir)
    
    asner_examples_dev= asner.get_examples('as','valid')
    asner_examples_train = asner.get_examples('as','train')
    asner_examples_train = asner.cleanify_tokens(asner_examples_train)
    asner_examples_test = asner.get_examples('as','test')
    asner_examples_test = asner.cleanify_tokens(asner_examples_test)
    
    print("Loaded Assamese NER Train, Test and Dev datasets successfully")
    
    label_list = []
    for i, j in enumerate(asner_examples_train):
        label_list.append(j.labels)
    flat_list = [item for sublist in label_list for item in sublist]

    label_list = set(flat_list)
     
          
          
    # Wiki NER datasets, just in case!       
    #ner = WikiNER(wiki_ner_dir)
    #wikiexamples_dev = ner.get_examples('as', 'as-valid')
    #wikiexamples_train = ner.get_examples('as', 'as-train')     
    device = torch.device('cuda:1')   
       #load models and tokenizers
    model_name="mbert"
    device_ids = [0]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    # tokenizer.model_max_length = 514
    #convert dataset into tokenized features and trim 
    
    features_train, features_dict_train, label_map_train = convert_tokens_examples_to_features(
    asner_examples_train,
    list(label_list),
    128,
    tokenizer,
    cls_token_at_end=False,
    cls_token='[CLS]',
    cls_token_segment_id=0,
    sep_token='[SEP]',
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=32000,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    full_token=False,
)

    features_dev, features_dict_dev, label_map_dev = convert_tokens_examples_to_features(
        asner_examples_dev,
        list(label_list),
        128,
        tokenizer,
        cls_token_at_end=False,
        cls_token='[CLS]',
        cls_token_segment_id=0,
        sep_token='[SEP]',
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=32000,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        full_token=False,
    )

    features_test, features_dict_test, label_map_test = convert_tokens_examples_to_features(
        asner_examples_test,
        list(label_list),
        128,
        tokenizer,
        cls_token_at_end=False,
        cls_token='[CLS]',
        cls_token_segment_id=0,
        sep_token='[SEP]',
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=32000,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        full_token=False,
    )


    print("ASNER tokens converted to features! ")
    #create label to ID and ID to label maps 

    label2id = label_map_train
    id2label =  {v: k for k, v in label2id.items()}
    label2id, id2label
    
    
    albert_model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-multilingual-cased', num_labels=6, id2label=id2label, label2id=label2id
    ).to(device)


    
    
    
 
    #initiate dataloader and trainer 
    
    train_dataloader = make_loader(features_train, 20)
    eval_dataloader = make_loader(features_dev,2)
    test_dataloader = make_loader(features_test, 2, False)
          
    optimizer = AdamW(albert_model.parameters(), lr=2e-5)
    num_train_epochs = 10
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
          
    output_dir = "mbert-ner-finetuned"     
     
          
    # train model for NER 
    progress_bar = tqdm.tqdm(range(num_training_steps))
    working_folder = '/s/carnap/b/nobackup/signal/m3x/axberta/task_finetune/mbert'   


 
    device = torch.device('cuda:1')
    total_predictions = []
    total_true  = []
    results = []
    training_loss = []
    val_results = []
    test_results = []
    # precision, recall, f_score, true_sum, accuracy
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_f1 = 0.0 

    eval_loaders = {
        'val': [eval_dataloader, val_results],
        'test': [test_dataloader, test_results]
    }

    '''
    need to properly save all metrics with epochs
    need to add loop for test loader
    '''
    for epoch in range(num_train_epochs):

        # Training

        running_loss = 0.0

        albert_model.train()
        for batch in train_dataloader:
            inputs = inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'token_type_ids': batch[2].to(device), 'labels': batch[3].to(device)}
            outputs = albert_model(**inputs)
            loss = outputs.loss
            # print("batch training loss", loss)
            training_loss.append(loss)
            loss.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        
            progress_bar.set_description(f'Epoch {epoch}, Loss: {epoch_loss:.3f}, batch_loss: {loss.item():.3f}, Val Accuracy: {epoch_accuracy:.3f}, Val F1: {epoch_f1:.3f}')
            progress_bar.update(1)

            

        # Evaluation
        epoch_loss = running_loss / len(train_dataloader)
        progress_bar.set_description(f'Epoch {epoch}, Loss: {epoch_loss:.3f}, Val Accuracy: {epoch_accuracy:.3f}, Val F1: {epoch_f1:.3f}')


        scorer_folder = working_folder + f'/asner_mbert_tagger/chk_{epoch}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        

        running_accuracy = 0.0
        running_f1 = 0.0 
        albert_model.eval()
        for set in eval_loaders:
            [loader, res_table] = eval_loaders[set]
            all_predictions, all_labels = [], []
            curr_res = [epoch, 0, 0, 0, 0]
            progress_bar.set_description(f'Evaluating {set} set...')
            for batch in loader:
                with torch.no_grad():
                    inputs = inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'token_type_ids': batch[2].to(device), 'labels': batch[3].to(device)}
                    outputs = albert_model(**inputs)

                predictions = outputs.logits.argmax(dim=-1)
                labels = inputs['labels']
                all_predictions.append(predictions)
                all_labels.append(labels)
                # validation_loss.append(outputs.loss)
    
                
            all_predictions = torch.vstack(all_predictions)
            all_labels = torch.vstack(all_labels)

            batch_results, accuracy = compute_metrics(all_predictions, all_labels,id2label)
            if set  =='test':
                all_labels, all_predictions = postprocess(all_predictions, all_labels, id2label)
                

            curr_res[1] = batch_results[0]
            curr_res[2] = batch_results[1]
            curr_res[3] = batch_results[2]
            curr_res[4] = accuracy
            # curr_res[5] = outputs.loss.item()

            running_accuracy = accuracy
            running_f1 = batch_results[2]


            # for i in range(1, 6):
            #     curr_res[i] = curr_res[i] / len(loader) 
            
            res_table.append(curr_res)
            
            if set == 'val':
                epoch_accuracy = running_accuracy
                epoch_f1 = running_f1


            if set =='test':

                with open(working_folder + f'/asner_mbert_tagger/chk_{epoch}/{set}_predictions.csv', "w") as f:
                    wr = csv.writer(f)
                    wr.writerows(all_predictions)

                with open(working_folder + f'/asner_mbert_tagger/chk_{epoch}/{set}_labels.csv', "w") as f:
                    wr = csv.writer(f)
                    wr.writerows(all_labels)


        albert_model.save_pretrained(scorer_folder + '/mbert_ner')
        tokenizer.save_pretrained(scorer_folder + '/mbert_ner')

        progress_bar.set_description(f'Epoch {epoch}, Loss: {epoch_loss:.3f}, Val Accuracy: {epoch_accuracy:.3f}, Val F1: {epoch_f1:.3f}')


    val_df = pd.DataFrame(val_results, columns=['epoch', 'precision', 'recall', 'f1', 'accuracy'])
    val_df.to_csv(working_folder + '/val_results.csv', index=False)

    test_df = pd.DataFrame(test_results, columns=['epoch', 'precision', 'recall', 'f1', 'accuracy'])
    test_df.to_csv(working_folder + '/test_results.csv', index=False)

 
 
    
    
    
    
    
    
    
    
    
    
    
    