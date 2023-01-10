#Author: Abhijnan Nath, CSU Signal Lab.  

import torch.nn as nn
import torch
from transformers import *
from transformers import AutoModel, AutoTokenizer, AutoModelForMultipleChoice, AutoModelForTokenClassification

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
 
import numpy as np
import pyhocon
import os
from inspect import getfullargspec
from datasets import load_dataset, load_metric
model_name="/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500"
from transformers import (AutoModelWithLMHead, 
                          AutoTokenizer, 
                          BertConfig)
device = torch.device('cuda:0')


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
                question='',
                contexts=[item['sectionText'], item['sectionText'],
                          item['sectionText'], item['sectionText']],
#                 endings=['<m>'+ " " + item['titleA']+ " "+'</m>','<m>'+" "+item['titleB']+" "+'</m>','<m>'+" "+item['titleC']+" "+'</m>',
#                          '<m>'+" "+item['titleD']+" "+'</m>'],
                
                endings=[item['titleA'],item['titleB'],item['titleC'],
                        item['titleD']],
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
    features_dict = defaultdict(dict)
    count = 0
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            count+=1
            if example.question.find("_") != -1:
                print("no question cannot find")
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = text_a + " " + ending

            inputs = tokenizer(
                 
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation='longest_first',
                pad_to_max_length=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)
            
#             print("count", count)
#             features_dict[count][ending_idx]["input_ids"] = inputs["input_ids"] 
#             features_dict[count][ending_idx]["attention_mask"] = inputs["attention_mask"] 
#             features_dict[count][ending_idx]["token_type_ids"] = inputs["token_type_ids"] 
#             #features[count]["positions_ids"] = torch.arange(len(inputs['input_ids'])) 
#             features_dict[count][ending_idx]["position_ids"] = torch.arange(inputs["input_ids"].shape[-1]).expand(inputs["input_ids"].shape)

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

    return features, label_map
def convert_multiple_choice_examples_to_features_asqa(
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
            
  
    return features  

def make_loader(features, batch_size):
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
            batch_size=batch_size,shuffle=True,
        )


def compute_metrics(predictions, labels, id2label):

    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    ]

    scores = precision_recall_fscore_support_asner(true_labels, true_predictions, average='macro')
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

def val_dataloader(self):
    dev_features = self.load_features('dev')
    dataloader = self.make_loader(dev_features, self.hparams['eval_batch_size'])
    return dataloader

def test_dataloader(self):
    test_features = self.load_features('test')
    dataloader = self.make_loader(test_features, self.hparams['eval_batch_size'])
    return dataloader





def compute_metrics(predictions, label_ids):
     
    #preds = np.argmax(predictions, axis=1)

    return {"accuracy": (predictions == label_ids).astype(np.float32).mean().item()}


def mean_accuracy(preds, labels):
    return {'acc': (preds == labels).mean()}

def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)

def flatten(l):
    result = []
    for element in l:
        if hasattr(element, "__iter__") and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

if __name__ == '__main__':
    
    #load the model and the tokenizer into device CUDA
    
    albert_model = AlbertForMultipleChoice.from_pretrained(model_name).to(device)

    tokenizer = AlbertTokenizer.from_pretrained("/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/")
    tokenizer.model_max_length = 514
    
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
        list(as_wiki_dev) ,
        tokenizer,
        512,
        list(label_list),
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) 


    test_features,test_label_map = convert_multiple_choice_examples_to_features(
        list(as_wiki_test) ,
        tokenizer,
        512,
        list(label_list),
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) 

    len(train_features), len(dev_features), len(test_features)

    #initiate dataloader and trainer 

    train_dataloader = make_loader(train_features, 40)
    eval_dataloader = make_loader(dev_features,20)
    test_dataloader = make_loader(test_features,20)

    optimizer = AdamW(albert_model.parameters(), lr=1e-6)
    num_train_epochs =30
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    output_dir = "axbert-ner-finetuned"    


    # train model for NER 
    progress_bar = tqdm.tqdm(range(num_training_steps))
    working_folder = '/s/chopin/d/proj/ramfis-aida/axbert/'   



    device = torch.device('cuda:0')
    total_predictions = []
    total_true  = []
    results = []
    training_loss = []
    validation_loss = []
    results = []
    eval_results = []
    val_accuracy = defaultdict(dict)
    epoch_true = []
    epoch_pred = []
    for epoch in range(num_train_epochs):

        # Training
        albert_model.train()
        for batch in train_dataloader:
            inputs =  {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'token_type_ids': batch[2].to(device), 'labels': batch[3].to(device)}
            outputs = albert_model(**inputs)
            loss = outputs.loss
            print("batch training loss", loss)
            training_loss.append(loss)
            loss.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            print(f'Iteration {epoch} Loss:', loss / len(train_dataloader))

        # Evaluation


        albert_model.eval()
        print("epoch", epoch)
        for batch in eval_dataloader:
            with torch.no_grad():
                #print("eval prediting acc")

                inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'token_type_ids': batch[2].to(device), 'labels': batch[3].to(device)}
                outputs = albert_model(**inputs)
               
                
            print("logits",outputs.logits )
            predictions = outputs.logits.argmax(dim=-1)
            predictions  = np.array(predictions.detach().cpu())
            
            
            validation_loss.append(outputs.loss)

            labels = np.array(inputs['labels'].detach().cpu())
            #print("outputs new predictions",predictions ), print("true labels", labels)
            total_predictions.append(predictions )
            total_true.append(labels)

            eval_results.append(accuracy(predictions, labels))
            print("BATCH DEV ACCURACY",sum(eval_results)/len(eval_results))
            val_accuracy[epoch] = sum(eval_results)/len(eval_results)
           
            #print(epoch,accuracy(predictions, labels))
             
            #val_acc = accuracy(predictions, labels)
        #val_acc  = eval_results[-4:-1]
        scorer_folder = working_folder + f'/aswiki_section/chk_{epoch}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        #model_path = scorer_folder + '/linear.chkpt'
        #torch.save(parallel_model.module.linear.state_dict(), model_path)
        albert_model.save_pretrained(scorer_folder + '/axbert_wiki')
        tokenizer.save_pretrained(scorer_folder + '/axbert_wiki')
        epoch_pred.append(total_predictions)
        epoch_true.append(total_true)
    
    a, b = flatten(epoch_pred[-1]), flatten(epoch_true[-1])
    print("FINAL EPOCH ACC", accuracy(np.array(a), np.array(b) ))    
        
    scorer_folder = working_folder + f'/aswiki_section/chk_{epoch}'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    #model_path = scorer_folder + '/linear.chkpt'
    #torch.save(parallel_model.module.linear.state_dict(), model_path)
    albert_model.save_pretrained(scorer_folder + '/axbert_wiki')
    tokenizer.save_pretrained(scorer_folder + '/axbert_wiki') 
    
    print("FINAL DEV ACCURACY",val_accuracy )

#             #print("dev results",eval_results 
#     #print("FINAL DEV ACCURACY",sum(eval_results)/len(eval_results) )
#     print("FINAL DEV ACCURACY",val_accuracy )
    

        

    
