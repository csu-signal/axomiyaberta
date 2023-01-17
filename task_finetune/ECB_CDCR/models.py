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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


    

    
class AxBERTa_pairwise(nn.Module):  
    def __init__(self, is_training=True, long=True, model_name=None,
                 linear_weights=None, pan=False,pan_features=None,max_pad_len=None):  #use the pretrained AxomiyaBERTa model from our google drive anonymous link provided in the github 
        super(AxBERTa_pairwise, self).__init__()


        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        
        #self.tokenizer =  XLMTokenizer.from_pretrained(model_name)
        self.long = long
        self.pan = pan
        self.max_pad_len = max_pad_len
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
#         if not self.long:
#             self.linear = nn.Sequential(
#                 nn.Linear(self.hidden_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
            
        if self.pan:

            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size + self.max_pad_len*2, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        else:

            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
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
        #cls_vector = last_hidden_states[:,0,:] #xlm last hidden state

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
            return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            #return torch.cat([cls_vector, arg2_vec], dim=1) # for AA, AB, AC triplet loss embeddings 
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector + arg1_vec +arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)
            #return torch.cat([cls_vector, arg1_vec * arg2_vec], dim=1)
            return cls_vector
        
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False,  panphon_features = None ):

        if pre_lm_out:
            return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
        lm_output = self.generate_model_output_cosalign(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
    
        if panphon_features is not None:
        
            lm_output = torch.cat([lm_output, panphon_features], dim=1)
        else:
            lm_output =lm_output 

        
        if lm_only:
            return lm_output
        out_scores = self.linear(lm_output.float()) # converted the lm output to float here because of the "RuntimeError: Expected object of scalar type Float but got scalar type Double"
     
        output = torch.sigmoid(out_scores)

        return  output,out_scores 
               
        #return lm_output
  

  class XLM_pairwise(nn.Module):  
    def __init__(self, is_training=True, long=True, model_name='xlm-mlm-en-2048',
                 linear_weights=None, pan=False,pan_features=None,max_pad_len=None):  #use the pretrained xlm-mlm-100-1280 model from huggingface, https://huggingface.co/xlm-mlm-100-1280
        super(XLM_pairwise, self).__init__()

      
        pan = False # we only use phonological features in the case of AxomiyaBERTa
        self.tokenizer =  XLMTokenizer.from_pretrained(model_name)
        self.long = long
        self.pan = pan
        self.max_pad_len = max_pad_len
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
#         if not self.long:
#             self.linear = nn.Sequential(
#                 nn.Linear(self.hidden_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
            
        if not self.pan:

            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size* 4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
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
        #cls_vector = last_hidden_states[:,0,:] #xlm last hidden state

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
            return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            #return torch.cat([cls_vector, arg2_vec], dim=1) # for AA, AB, AC triplet loss embeddings 
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector + arg1_vec +arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)
            #return torch.cat([cls_vector, arg1_vec * arg2_vec], dim=1)
            return cls_vector
        
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False,  panphon_features = None ):
        
        panphon_features = None
        if pre_lm_out:
            return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
    
        if panphon_features is not None:
        
            lm_output = torch.cat([lm_output, panphon_features], dim=1)
        else:
            lm_output =lm_output 

        
        if lm_only:
            return lm_output
        out_scores = self.linear(lm_output.float()) # converted the lm output to float here because of the "RuntimeError: Expected object of scalar type Float but got scalar type Double"
     
        output = torch.sigmoid(out_scores)

        return  output,out_scores 


  class IndicBERT_pairwise(nn.Module):  
    def __init__(self, is_training=True, long=True, model_name="ai4bharat/indic-bert",
                 linear_weights=None, pan=False,pan_features=None,max_pad_len=None):  #use the pretrained xlm-mlm-100-1280 model from huggingface, https://huggingface.co/xlm-mlm-100-1280
        super(IndicBERT_pairwise, self).__init__()

      
        pan = False # we only use phonological features in the case of AxomiyaBERTa
        self.tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/indic-bert")
        self.long = long
        self.pan = pan
        self.max_pad_len = max_pad_len
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
#         if not self.long:
#             self.linear = nn.Sequential(
#                 nn.Linear(self.hidden_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
            
        if not self.pan:

            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size* 4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
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
        #cls_vector = last_hidden_states[:,0,:] #xlm last hidden state

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
            return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            #return torch.cat([cls_vector, arg2_vec], dim=1) # for AA, AB, AC triplet loss embeddings 
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector + arg1_vec +arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)
            #return torch.cat([cls_vector, arg1_vec * arg2_vec], dim=1)
            return cls_vector
        
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False,  panphon_features = None ):
        
        panphon_features = None
        if pre_lm_out:
            return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
    
        if panphon_features is not None:
        
            lm_output = torch.cat([lm_output, panphon_features], dim=1)
        else:
            lm_output =lm_output 

        
        if lm_only:
            return lm_output
        out_scores = self.linear(lm_output.float()) # converted the lm output to float here because of the "RuntimeError: Expected object of scalar type Float but got scalar type Double"
     
        output = torch.sigmoid(out_scores)

        return  output,out_scores 



  class MuRIL_pairwise(nn.Module):  
    def __init__(self, is_training=True, long=True, model_name='google/muril-base-cased',
                 linear_weights=None, pan=False,pan_features=None,max_pad_len=None):  #use the pretrained xlm-mlm-100-1280 model from huggingface, https://huggingface.co/xlm-mlm-100-1280
        super(MuRIL_pairwise, self).__init__()

        model_name='google/muril-base-cased'
      
        pan = False # we only use phonological features in the case of AxomiyaBERTa
        self.tokenizer =  AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.long = long
        self.pan = pan
        self.max_pad_len = max_pad_len
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
#         if not self.long:
#             self.linear = nn.Sequential(
#                 nn.Linear(self.hidden_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
            
        if not self.pan:

            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size* 4, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
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
        #cls_vector = last_hidden_states[:,0,:] #xlm last hidden state

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
            return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)
            #return torch.cat([cls_vector, arg2_vec], dim=1) # for AA, AB, AC triplet loss embeddings 
        
    def generate_model_output_cosalign(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                      global_attention_mask, arg1, arg2)
        
        #print(arg1_vec)
        if not self.long:
            return cls_vector
        else:
            #return torch.cat([cls_vector + arg1_vec +arg2_vec], dim=1)
            #return torch.cat((cls_vector * arg1_vec * arg2_vec).sum(1), dim=1)
            #return torch.cat([cls_vector, arg1_vec * arg2_vec], dim=1)
            return cls_vector
        
                             
      
    def frozen_forward(self, input_):
        return self.linear(input_)

    
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False,  panphon_features = None ):
        
        panphon_features = None
        if pre_lm_out:
            return self.linear(input_ids)

#         lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
#                                                global_attention_mask=global_attention_mask,
#                                                position_ids=position_ids,
#                                                arg1=arg1, arg2=arg2)
        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        
    
        if panphon_features is not None:
        
            lm_output = torch.cat([lm_output, panphon_features], dim=1)
        else:
            lm_output =lm_output 

        
        if lm_only:
            return lm_output
        out_scores = self.linear(lm_output.float()) # converted the lm output to float here because of the "RuntimeError: Expected object of scalar type Float but got scalar type Double"
     
        output = torch.sigmoid(out_scores)

        return  output,out_scores 