

import torch.nn as nn
import torch
from transformers import *
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pyhocon
import os
from inspect import getfullargspec

from transformers import (AutoModelWithLMHead, 
                          AutoTokenizer, 
                          BertConfig)
# relative path of config file

config_file_path = os.path.dirname(__file__) + '/../cdlm/config_pairwise_long_reg_span.json'
config = pyhocon.ConfigFactory.parse_file(config_file_path)


# print(config.cdlm_path)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)



    

    
class AxBERTa_Triplet(nn.Module):
    def __init__(self, is_training=True, long=True, model_name="/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500",
                 linear_weights=None):
        super(AxBERTa_Triplet, self).__init__()
        self.tokenizer = AlbertTokenizer.from_pretrained("/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/")
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            #self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]
        
        self.docstart_id = self.tokenizer.encode('<doc-s>', add_special_tokens=False)[0]
        self.docend_id = self.tokenizer.encode('</doc-s>', add_special_tokens=False)[0]
        
        self.vals = [self.start_id, self.end_id]
        self.docvals = [self.docstart_id, self.docend_id]
        self.hidden_size = self.model.config.hidden_size
        if not self.long:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
                 # for getting the same dimension as CDLM or longformer for cosine sim alignment 
            )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if 'global_attention_mask' in arg_names:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=global_attention_mask)
        else:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            return torch.cat([cls_vector, arg2_vec], dim=1) # for AA, AB, AC triplet loss embeddings 
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            return torch.cat([cls_vector + arg1_vec +arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)       
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

#     def forward(self, input_ids, attention_mask=None, position_ids=None,
#                 global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

#         if pre_lm_out:
#             return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
#         if lm_only:
#             return lm_output

#         return self.linear(lm_output)
    
    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
        
        if lm_only:
            return lm_output

        return self.linear(lm_output) 
        #return lm_output
    
class AxBERTa_EmbeddingDisperser(nn.Module):
    def __init__(self, is_training=True, long=True, model_name="/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/checkpoint-485500",
                 linear_weights=None):
        super(AxBERTa_EmbeddingDisperser, self).__init__()
        self.tokenizer = AlbertTokenizer.from_pretrained("/s/chopin/d/proj/ramfis-aida/loan_exp_results/loan-word-detection/Datasets/Assamese_Bert_dataset/data_dir_final/")
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            #self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]
        
        self.docstart_id = self.tokenizer.encode('<doc-s>', add_special_tokens=False)[0]
        self.docend_id = self.tokenizer.encode('</doc-s>', add_special_tokens=False)[0]
        
        self.vals = [self.start_id, self.end_id]
        self.docvals = [self.docstart_id, self.docend_id]
        self.hidden_size = self.model.config.hidden_size
        if not self.long:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.embed = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                #nn.Tanh(),
                #nn.Linear(128, 1),
                #nn.Sigmoid()
            )
            


            self.option = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 128)
             # for getting the same dimension as CDLM or longformer for cosine sim alignment 
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
            self.embed.apply(init_weights)
            self.option.apply(init_weights)
            
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if 'global_attention_mask' in arg_names:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=global_attention_mask)
        else:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1) # arg 1 and arg1 for [MASK] and the option token respectively
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            return torch.cat([cls_vector, arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)       
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

#     def forward(self, input_ids, attention_mask=None, position_ids=None,
#                 global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

#         if pre_lm_out:
#             return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
#         if lm_only:
#             return lm_output

#         return self.linear(lm_output)
    
    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
        option_token_output= self.generate_model_output_cosalign(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        if lm_only:
            return self.option(option_token_output)

        return self.linear(lm_output), self.embed(lm_output)
        #return lm_output
