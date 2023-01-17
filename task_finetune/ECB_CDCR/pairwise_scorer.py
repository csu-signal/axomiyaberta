import os
import sys
import gc
import numpy as np
import pandas as pd
import csv
gc.collect()
import torch
import torch.nn as nn
import torch.nn as F
from collections import defaultdict
from collections import Counter
#torch.cuda.empty_cache()
print(torch.cuda.current_device())
parent_path = "../../"
sys.path.append(parent_path)

import os.path
import pickle

from sklearn.model_selection import train_test_split
import pyhocon
from models import AxBERTa_pairwise
 
import random
from tqdm.autonotebook import tqdm
#from parsing.parse_ecb import parse_annotations
import epitran
import panphon
 

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
        
            mention_pair = line[:2]
   
            if line[2] =='POS':
                triviality_label = 1
                all_examples.append((mention_pair, triviality_label))
           
                
            else:
                triviality_label = 0
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


def tokenize(tokenizer, mention_pairs, mention_map,m_start, m_end, max_sentence_len=128, context = "bert_doc"):
 
    if max_sentence_len is None:
        
        max_sentence_len = tokenizer.model_max_length #try 512 here, 
        
    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []
    pan_1 = []
    pan_2 = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
        
 
        if context =="bert_doc":

            sentence_a = mention_map[m1]['bert_doc_assamese'].replace("[", "").replace("]","")
            #print("Dev set sentence A", sentence_a)
            sentence_b = mention_map[m2]['bert_doc_assamese'].replace("[", "").replace("]","")
            
            #print("Dev set sentence B", sentence_b)
        else:
            sentence_a = mention_map[m1]['bert_sentence'] 
            sentence_b = mention_map[m2]['bert_sentence'] 

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]),                    ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)
        pan_1.append(mention_map[m1]['pan_features'])
        pan_2.append(mention_map[m2]['pan_features'])
        

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        
        if context == "bert_sentence":
            
            for input_id in input_ids:
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
                in_truncated = input_id[m_end_index-(max_sentence_len//4): m_end_index] +                                input_id[m_end_index: m_end_index + (max_sentence_len//4)]


                in_truncated = in_truncated + [tokenizer.pad_token_id]*(max_sentence_len//2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)
        else:               
            for input_id in input_ids:
    
                global_input_id = [1]
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)

                doc_start_token = [32001]
                doc_end_token =  [32002]
                curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

                in_truncated = input_id[curr_start_index: m_end_index] + \
                               input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
                in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances, pan_1,pan_2):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))
        
        #pan_a, pan_b = zip(*panphon_features)
        pan_a, pan_b = pan_1,pan_2
        pan_a = torch.stack(pan_a, dim =0).squeeze()
        pan_b = torch.stack(pan_b, dim =0).squeeze()
        #print(pan_a), print(pan_b)
        pan = torch.hstack((pan_a, pan_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab,
                             'pan_features':pan
                             }

        return tokenized_ab_dict
    
    
    #get the panphon features and 
    tokenized_ab = ab_tokenized(pairwise_bert_instances_ab, pan_1,pan_2)
    #tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)

    return tokenized_ab 

def tokenize_triplets(tokenizer, mention_triplets, mention_map,m_start, m_end, max_sentence_len=512, context = "bert_doc"):
 
    if max_sentence_len is None:
        
        max_sentence_len = tokenizer.model_max_length #try 512 here, 
        
    #max_sentence_len=2048 #trying out a greater context since Longformer! 

    pairwise_bert_instances_aa = []
    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ac = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2, m3) in mention_triplets:
        
         if context =="bert_doc":

            sentence_a = mention_map[m1]['bert_doc_assamese'].replace("[", "").replace("]","")
            #print("Train set sentence A", sentence_a )
            sentence_b = mention_map[m2]['bert_doc_assamese'].replace("[", "").replace("]","")
            #print("Train set sentence B", sentence_b )
            sentence_c = mention_map[m3]['bert_doc_assamese'].replace("[", "").replace("]","")
            #print("Tran set sentence C", sentence_c )
        else:
            #print(m1, m2, m3)
            sentence_a = mention_map[m1]['bert_sentence'] 
            #print("sentence A", sentence_a )
            sentence_b = mention_map[m2]['bert_sentence'] 
            #print("sentence B", sentence_b )
            sentence_c = mention_map[m3]['bert_sentence'] 
            #print("sentence C", sentence_c )
            

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]),                    ' '.join([doc_start, sent_b, doc_end])

        
        instance_aa = make_instance(sentence_a, sentence_a)
        
        pairwise_bert_instances_aa.append(instance_aa)
       # print("sentence aa",len(instance_aa))
        
        instance_ab = make_instance(sentence_a, sentence_b)
        #print("sentence ab",instance_ab)
        pairwise_bert_instances_ab.append(instance_ab)
        
        instance_ac = make_instance(sentence_a, sentence_c)
        #print("sentence ac", instance_ac)
        pairwise_bert_instances_ac.append(instance_ac)

#         instance_ba = make_instance(sentence_b, sentence_a)
#         pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        
        if context == "bert_sentence":
            
            for input_id in input_ids:
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
                in_truncated = input_id[m_end_index-(max_sentence_len//4): m_end_index] +                                input_id[m_end_index: m_end_index + (max_sentence_len//4)]


                in_truncated = in_truncated + [tokenizer.pad_token_id]*(max_sentence_len//2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)
        else:            
            #print(context)
            for input_id in input_ids:
                #print(input_id)
                #global_input_id = torch.ones([1,1])
                global_input_id = [1]
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
 
                doc_start_token = [32001]
                doc_end_token =  [32002]
            
                curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

                in_truncated = input_id[curr_start_index: m_end_index] + \
                               input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
                in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b,  = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        
        #tokenized_a = tokenizer(instances_a, add_special_tokens=False,padding='max_length', return_tensors="pt")
        
        #tokenized_b = tokenizer(instances_b, add_special_tokens=False,padding='max_length', return_tensors="pt")
        
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    tokenized_aa = ab_tokenized(pairwise_bert_instances_aa)
    tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
    tokenized_ac = ab_tokenized(pairwise_bert_instances_ac)
    #tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)

    #return tokenized_ab, tokenized_ba
    return tokenized_aa, tokenized_ab, tokenized_ac

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


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False, pan=False ):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    batch_pan_ab = ab_dict['pan_features'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)
    
    if not pan:
         return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only, panphon_features = None)
        

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only, panphon_features = batch_pan_ab) 


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




def cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    cos_dev = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
     

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            #dev_pairs = dev_pairs [batch_indices ]
                                          
            cos_ab = forward_ab(parallel_model, dev_ab, device, batch_indices) # these scores are actually embeddings 
            cos_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            batch_predictions = cos_dev(cos_ab, cos_ba).detach().cpu()
            print("batch dev prediction tensor", batch_predictions)
            batch_predictions = (batch_predictions > .95).detach().cpu()
       
            predictions.append(batch_predictions)

    return torch.cat(predictions)


def predict(parallel_model, device, dev_ab, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    scores = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab, scores_pair = forward_ab(parallel_model, dev_ab, device, batch_indices, pan=True) 
            #print("dev pairwise scores", scores_pair)
            batch_predictions = (scores_ab > 0.5).detach().cpu()
            predictions.append(batch_predictions)
            scores.append(scores_pair )
            
    
    return torch.cat(predictions), torch.cat(scores)


def flatten(l):
    result = []
    for element in l:
        if hasattr(element, "__iter__") and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result



def find_panpad_maxlength(train, valid, test):
    
    #train
    ft = panphon.FeatureTable()
   
    list_words_train = []
    list_words_valid = []
    list_words_test = []
   

    for x, y in train.items():
        
    
        list_words_train.append(y['assamese_lemma'])

    df_train= pd.DataFrame()
    df_train['words'] = list_words_train   

    epi = epitran.Epitran('asm-Beng')
    
    df_train['features_train_epi'] = df_train.apply(lambda x:epi.transliterate(x['words']), axis=1 )
    df_train['features_train'] = df_train.apply(lambda x:ft.word_to_vector_list(x['features_train_epi'],numeric=True ), axis=1)
    df_train['features_train'] = df_train['features_train'].apply(lambda x:sum(x, []))
    max_len_train = 24*(df_train.features_train_epi.map(lambda x: len(ft.word_to_vector_list(x))).max())
    
    #dev
        
    for x, y in valid.items():
         
        list_words_valid.append(y['assamese_lemma'])

    df_valid= pd.DataFrame()
    df_valid['words'] = list_words_valid 

    epi = epitran.Epitran('asm-Beng')
 
    df_valid['features_train_epi'] = df_valid.apply(lambda x:epi.transliterate(x['words']), axis=1 )
    df_valid['features_train'] = df_valid.apply(lambda x:ft.word_to_vector_list(x['features_train_epi'],numeric=True ), axis=1)
    df_valid['features_train'] = df_valid['features_train'].apply(lambda x:sum(x, []))
    max_len_dev = 24*(df_valid.features_train_epi.map(lambda x: len(ft.word_to_vector_list(x))).max())
    
    #test
    
    for x, y in test.items():
      
        list_words_test.append(y['assamese_lemma'])

    df_test= pd.DataFrame()
    df_test['words'] = list_words_test 

    epi = epitran.Epitran('asm-Beng')
    #sample = wikiexamples_train 
    df_test['features_train_epi'] = df_test.apply(lambda x:epi.transliterate(x['words']), axis=1 )
    df_test['features_train'] = df_test.apply(lambda x:ft.word_to_vector_list(x['features_train_epi'],numeric=True ), axis=1)
    df_test['features_train'] = df_test['features_train'].apply(lambda x:sum(x, []))
    max_len_test = 24*(df_test.features_train_epi.map(lambda x: len(ft.word_to_vector_list(x))).max())
    
    max_list = [max_len_train, max_len_dev,  max_len_test]
    print("split wise max", max_list)
    max_overall = max(max_list)
    
    
    return max_overall
    
 
def BCELoss_class_weighted(input, target, weights):
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
    return torch.mean(bce)

    
def create_panphon_features(examples, pan_max_len):
    
    df_train= pd.DataFrame()
    list_words_train = []
   
    for x, y in examples.items():
       
        list_words_train.append(y['assamese_lemma'])

    df_train= pd.DataFrame()
    df_train['words'] = list_words_train   

    epi = epitran.Epitran('asm-Beng')
    ft = panphon.FeatureTable()
    
    
    df_train['features_train_epi'] = df_train.apply(lambda x:epi.transliterate(x['words']), axis=1 )
    df_train['features_train'] = df_train.apply(lambda x:ft.word_to_vector_list(x['features_train_epi'],numeric=True ), axis=1)
    df_train['features_train'] = df_train['features_train'].apply(lambda x:sum(x, []))
    max_len_train = 24*(df_train.features_train_epi.map(lambda x: len(ft.word_to_vector_list(x))).max())
    
    
    print("max length to be padded for Pan features", pan_max_len)

    df_train['features_train'] = df_train['features_train'].apply(lambda x: \
                                        np.pad(x,\
                                        (0,pan_max_len-len(x)), 'constant'))
    
  
    for i, (x, y) in enumerate(examples.items()):
        examples[x]['pan_features'] = torch.tensor(df_train['features_train'][i] )
            
    #return [x for x in df_train['features_train'] ] 
    return examples
    
    

def train_pairwise(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map_train,
          mention_map_dev,
                  
          mention_map_test,
                  
          working_folder,
          device,
          batch_size=32,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    bce_weights = torch.tensor([1.1,20.1])
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])



    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)
    
    dev_labels = torch.LongTensor(dev_labels)
       
    new_batch_size = batch_size
    chunk_size =5000
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42)
    train_indices = list(range(len(train_pairs)))
    random.Random(42).shuffle(train_indices)

    train_pairs = list((train_pairs[i] for i in train_indices))
    train_labels = list((train_labels[i] for i in train_indices))
    train_labels = torch.FloatTensor(train_labels)
    
    dev_aa  = tokenize(tokenizer, dev_pairs, mention_map_dev, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    
    for n in range(n_iters):
        random.Random(42).shuffle(train_indices)
        train_pairs = list((train_pairs[i] for i in train_indices))
        train_labels = list((train_labels[i] for i in train_indices))
        train_labels = torch.FloatTensor(train_labels)
  
        iteration_loss = 0.

        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()

            chunk_train_indices = train_indices[i: i + new_batch_size]

            chunk_train_pairs= train_pairs[i: i + new_batch_size]

            batch_indices = list(range(len(chunk_train_indices)))
     
            train_aa = tokenize(tokenizer, chunk_train_pairs, mention_map_train,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
           
            dev_pairs = dev_pairs[i: i + new_batch_size]

            mini_batch_indices = train_indices[i: i + new_batch_size]

            sig_scores , scores= forward_ab(parallel_model, train_aa, device, batch_indices, pan=True)            
            batch_labels = train_labels[i: i + new_batch_size].reshape((-1, 1)).to(device)
            loss = bce_loss(sig_scores, batch_labels) 
            loss = BCELoss_class_weighted(sig_scores, batch_labels, bce_weights) 

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))

        dev_predictions, scores = predict(parallel_model, device,dev_aa,batch_size)
        
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))
               
        scorer_folder = working_folder + f'/pairwise_scorer_axb/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear.chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/pairwise_scorer_axb/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
 
def batching(n, batch_size, min_batch):
    new_batch_size = batch_size
    while n % new_batch_size < min_batch:
        new_batch_size -= 1
    return new_batch_size

def predict_trained_model_tp_fp(parallel_model,mention_map,  path_tp, path_tn, working_folder, batch_size=500):
    parallel_model.eval()
    tp_pairs, tp_labels = zip(*load_data_tp_fp(path_tp))
    tn_pairs, tn_labels = zip(*load_data_tp_fp(path_tn))
    
    all_test_labels =tp_labels+ tn_labels
    all_test_pairs = tp_pairs + tn_pairs
    print(len(all_test_pairs)), print(len(all_test_labels))
    
    test_labels = torch.LongTensor(all_test_labels)
    
    all_test_pairs = all_test_pairs 
          
    device = torch.device('cuda:1')

    device_ids = [1]


    tokenizer = parallel_model.module.tokenizer
    
    n = len(all_test_pairs)
    indices = list(range(n))
    predictions = []
    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):

            batch_pairs = all_test_pairs[i: i + batch_size]

            batch_indices = list(range(len(batch_pairs)))

            test_ab = tokenize(tokenizer, batch_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
            

            scores_ab, scores_pair = forward_ab(parallel_model, test_ab, device, batch_indices,  pan=True) 
     
            batch_predictions = (scores_ab > 0.5).detach().cpu()
            predictions.append(scores_ab,)
            scores.append(scores_pair )
          

    predictions = torch.cat(predictions)
    scores = torch.cat(scores)
 
    return scores,predictions, test_labels,all_test_pairs

def get_singletons(mention_map):
       
    gold_cluster = []

    for i, j in mention_map.items():

        gold_cluster.append(j['gold_cluster'])

    singleton = dict(Counter(gold_cluster))
    single = []
    for i, j in singleton.items():
        if j ==1:
            single.append(i)


    men1 = []
    c=0
    for i, j in mention_map.items():
        if j['gold_cluster'] in single:
            men1.append(j['mention_id'])

    singleton_pairs = [[a,a] for a in men1]
    singleton_label = [(1) for a in men1]
    return singleton_pairs, singleton_label

def create_train_sample(singleton_pairs, singleton_label):
    
    working_folder = "../../As_Indic_data/ecb/"
 
    
    triv_train_path = working_folder + '/lemma_balanced_tn_fn_train.tsv'
    triv_dev_path = working_folder + '/lemma_balanced_tn_fn_dev.tsv'
    triv_test_path = working_folder + '/lemma_balanced_tn_fn_test.tsv'

    triv_train_path1 = working_folder + '/lemma_balanced_tp_fp_train.tsv'
    triv_dev_path1 = working_folder + '/lemma_balanced_tp_fp_dev.tsv'
    triv_test_path1 = working_folder + '/lemma_balanced_tp_fp_test.tsv'

    dev_pairs, dev_labels= zip(*load_data_tp_fp(triv_dev_path))
    train_pairs, train_labels= zip(*load_data_tp_fp(triv_train_path))
    test_pairs, test_labels= zip(*load_data_tp_fp(triv_test_path))

    dev_pairs1, dev_labels1= zip(*load_data_tp_fp(triv_dev_path1))
    train_pairs1, train_labels1= zip(*load_data_tp_fp(triv_train_path1))
    test_pairs1, test_labels1= zip(*load_data_tp_fp(triv_test_path1))


    trim_pos = []
    trim_neg = []
    
    #down sample the true negative pairs from the train set during training for smoother convergence

    for i, j in zip(train_pairs, train_labels):



        if j==1:
            trim_pos.append(i)
        else:
            trim_neg.append(i)

    trim_pairs=trim_pos[:7023] + trim_neg[0:80000]
    trim_labels= [1]*7023 + [0]*len(trim_neg[0:80000])

    #singleton_pairs, singleton_label = get_singletons(train_mention_map)
    new_train_labels = list(train_labels1) + trim_labels + singleton_label  
    new_train_pairs= list(train_pairs1)+trim_pairs + singleton_pairs 
    
    return new_train_pairs, new_train_labels 


def get_assamese_lemma(test_mention_map):
    
    c = '<m>'
    d = '</m>'
    for i, j in list(test_mention_map.items()) :

       
        s = j['bert_doc_assamese']
        s_split = str(j['bert_doc_assamese']).split()

        start = [pos for pos, char in enumerate(str(s).split()) if char == c ]
        end =[pos for pos, char in enumerate(str(s).split()) if char == d ]

        if len(s_split[ start[0]+1: end[0]])>1:
            mention_text = s_split[ start[0]+1: end[0]][0]
     
            if isinstance(mention_text, list):
            

                test_mention_map[i]['assamese_lemma'] = mention_text[0]
            else:
                test_mention_map[i]['assamese_lemma'] = mention_text 


        elif len(s_split[ start[0]+1: end[0]])==1:

            mention_text = s_split[ start[0]+1: end[0]]

            if isinstance(mention_text, list):
   
                test_mention_map[i]['assamese_lemma'] = mention_text[0]
            else:
                test_mention_map[i]['assamese_lemma'] = mention_text 
        elif len(s_split[ start[0]+1: end[0]])==0:
            mention_text = s_split[ end[0]+1: end[0]+4][0]  #find nearest word
       
            test_mention_map[i]['assamese_lemma'] = mention_text 
            
    return test_mention_map
        

if __name__ == '__main__':
        
    parent_path = "../../"
     
    working_folder = parent_path + "/As_Indic_data/ecb/"
   
    triv_train_path = working_folder + '/lemma_balanced_tn_fn_train.tsv'
    triv_dev_path = working_folder + '/lemma_balanced_tn_fn_dev.tsv'
    triv_test_path = working_folder + '/lemma_balanced_tn_fn_test.tsv'
   

    dev_pairs, dev_labels= zip(*load_data_tp_fp(triv_dev_path))
    
    dev_pos = []
    dev_neg = []

    for i, j in zip(dev_pairs,dev_labels):

        if j==1:
            dev_pos.append(i)
        else:
            dev_neg.append(i)

    dev_pairs=dev_pos[:50] + dev_neg[0:500]
    dev_labels= [1]*50 + [0]*len(dev_neg[0:500])

 
    working_folder = "../../As_Indic_data/ecb/"
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    

    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key
        
     
    with open( working_folder+ "/ass_trans_dict_next_train.pkl", "rb") as fp:   # Unpickling longformer scores
        score_map_train = pickle.load(fp)
 
    with open(working_folder + "/ass_trans_dict_next_dev.pkl", "rb") as fp:   # Unpickling longformer scores
        score_map_dev = pickle.load(fp)

    with open(working_folder+ "/ass_trans_dict_next_test.pkl", "rb") as fp:   # Unpickling longformer scores
        score_map_test = pickle.load(fp)
    
    # create train assamese ECB mention map     
    train_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == "evt" and
                                                        val['split'] == "train"}
    print(len(train_mention_map), len(score_map_train))
    for i, j in train_mention_map.items():

        if i in score_map_train.keys():
            men_doc = str(score_map_train[i]).replace('</m>',' </m> ').replace('<m>',' <m> ')
            train_mention_map[i]['bert_doc_assamese'] = men_doc

    # create dev assamese ECB mention map   
    dev_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == "evt" and
                                                        val['split'] == "dev"}
    print(len(dev_mention_map), len(score_map_dev))
    for i, j in dev_mention_map.items():

        if i in score_map_dev.keys():
            men_doc = str(score_map_dev[i]).replace('</m>',' </m> ').replace('<m>',' <m> ')
            dev_mention_map[i]['bert_doc_assamese'] = men_doc
    
    # create TEst assamese ECB mention map   
    test_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == "evt" and
                                                        val['split'] == "test"}
    print(len(test_mention_map), len(score_map_test))
    for i, j in test_mention_map.items():

        if i in score_map_test.keys():
            men_doc = str(score_map_test[i]).replace('</m>',' </m> ').replace('<m>',' <m> ')
            test_mention_map[i]['bert_doc_assamese'] = men_doc
             
    
    #get the assamese lemma for the trigger events from the documents
    train_mention_map = get_assamese_lemma(train_mention_map)
    dev_mention_map = get_assamese_lemma(dev_mention_map)
    test_mention_map = get_assamese_lemma(test_mention_map)
       
    #find the max length of all the splits together:
       
    maxlen = find_panpad_maxlength(train_mention_map, dev_mention_map, test_mention_map) 
    print("maxlen", maxlen)     
    working_folder = parent_path + "/As_Indic_data/ecb/"
    
    scorer_folder = working_folder +'/pairwise_scorer_axberta_pan_upsample_arg/chk_8/'
    model_name = scorer_folder + 'bert'
    linear_weights_path = scorer_folder + 'linear.chkpt'
  
    device = torch.device('cuda:1')
#     device = torch.device("cpu")
    
    print("deviceID ", device)
   
    # LOAD AxomiyaBERTa model and the tokenizer files from our google drive anonymous link 
    # For running other baselines, replace the model_name with the appropriate pretrained model 
    
    
    model_name = model_name #Load from Google Drive here! 
   
#     linear_weights_path = 'linearcdlm'
#     linear_weights = torch.load(linear_weights_path)

    scorer_module = AxBERTa_pairwise(is_training=True, model_name=model_name, linear_weights=None,  pan=True,pan_features=None,max_pad_len=maxlen).to(device)
   

#     #device_ids = list(range(2))
    device_ids = [1]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)

    parallel_model.module.to(device)


    train_mention_map_path = working_folder + '/train_mention_map_ecb.pkl'
    dev_mention_map_path = working_folder + '/dev_mention_map_ecb.pkl'
    test_mention_map_path = working_folder + '/test_mention_map_ecb.pkl'
    
    if not os.path.exists(train_mention_map_path) :
        train_mention_map = create_panphon_features(train_mention_map, maxlen)
        pickle.dump(train_mention_map, open(train_mention_map_path, 'wb'))
    else:
        train_mention_map = pickle.load(open(train_mention_map_path, 'rb'))

        
    if not os.path.exists(dev_mention_map_path) :
        dev_mention_map = create_panphon_features(dev_mention_map, maxlen)
        pickle.dump(dev_mention_map, open(dev_mention_map_path, 'wb'))
    else:
        dev_mention_map = pickle.load(open(dev_mention_map_path, 'rb'))
    
    if not os.path.exists(test_mention_map_path) :
        test_mention_map = create_panphon_features(test_mention_map, maxlen)
        pickle.dump(test_mention_map, open(test_mention_map_path, 'wb'))
    else:
        test_mention_map = pickle.load(open(test_mention_map_path, 'rb'))

    
    #create the pan features for each split 
    mention_map = dict(train_mention_map)
    mention_map.update(test_mention_map)
    mention_map.update(dev_mention_map)
    

    device_ids = [1]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    #print(parallel_model)
    working_folder = parent_path + "/parsing/ecb"
    
    singleton_pairs, singleton_label = get_singletons(train_mention_map)
    train_pairs, train_labels = create_train_sample(singleton_pairs, singleton_label)
      
    train_pairwise(train_pairs[0:40],
          train_labels[0:40],
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          mention_map,
                  
          test_mention_map,
                  
          working_folder,
          device,
          batch_size=4,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001)

    path_tn = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_dev.tsv'
    path_tp = parent_path + '/parsing/ecb/lemma_balanced_tp_fp_dev.tsv'
    
    
# GET DEV pairwise affinity scores after training and save them     
    
    scores,predictions, test_labels,all_test_pairs = predict_trained_model_tp_fp(parallel_model,dev_mention_map, path_tp, path_tn, working_folder, batch_size=1000)
    print(len(scores)), print(len(test_labels)), print(len(all_test_pairs))
    
    scores_folder = working_folder + f'/axbpan_dev_scores/'
    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)

    pickle.dump(scores, open(scores_folder + '/dev_prediction_scores', 'wb'))
    pickle.dump(predictions, open(scores_folder + '/dev_prediction_sigmoids', 'wb'))
    pickle.dump(test_labels, open(scores_folder + '/dev_labels', 'wb'))
    pickle.dump(all_test_pairs, open(scores_folder + '/dev_pairs', 'wb'))
    
# GET TEST pairwise affinity scores after training and save them         
    
    path_tn = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_test.tsv'

    path_tp = parent_path + '/parsing/ecb/lemma_balanced_tp_fp_test1.tsv'

    scores,predictions, test_labels,all_test_pairs = predict_trained_model_tp_fp(parallel_model,test_mention_map, path_tp, path_tn, working_folder,  batch_size=500)
    print(len(scores)), print(len(test_labels)), print(len(all_test_pairs))
    scores_folder = working_folder + f'/axbpan_test_scores/'
    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)

    pickle.dump(scores, open(scores_folder + '/test_prediction_scores', 'wb'))
    pickle.dump(predictions, open(scores_folder + '/test_prediction_sigmoids', 'wb'))
    
    pickle.dump(test_labels, open(scores_folder + '/test_labels', 'wb'))
    pickle.dump(all_test_pairs, open(scores_folder + '/test_pairs', 'wb'))
    
    

    
    

    

 


 








 
