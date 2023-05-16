# AxomiyaBERTa: A Phonologically-aware Transformer Model for Assamese
Abhijnan Nath, Sheikh Mannan, and Nikhil Krishnaswamy\
Situated Grounding Natural Language (SIGNAL) Lab, Colorado State University, Fort Collins, CO, USA\
Findings of the Association of Computational Linguistics: ACL 2023, Toronto, ON, Canada\
July 2023
 
This repo contains code for training, fine-tuning and evaluating a smaller-sized monolingual language model for Assamese along with its phonological versions. AxomiyaBERTa is a novel, mono-lingual Transformer language model for the Assamese language which has been trained in a low-resource and limited-compute setting, using only the masked language modeling (MLM) objective. Beyond a model for a new language, our novel contributions are as follows:

- Use of a novel combined loss technique to disperse AxomiyaBERTa’s embeddings;
- Addition of phonological articulatory features as an alternate performance improvement in the face of omitting the NSP training objective for longer-context tasks;
- Evaluation on event coreference resolution, which is novel for Assamese.

AxomiyaBERTa achieves competitive or state of the art results on multiple tasks, and demonstrates their utility for building new language models in resource-constrained settings.

## Structure 
The repository contains:
- `As_Indic_data` contains all the dataset files for each task along with auxillary files like meta-data for cross-document coreference resolution for Assamese ECB+ corpus along with the phonological-features for the two multiple choice tasks: `wiki-section title prediction` and `CLoze-QA`.
- Implementation of the AxomiyaBERTa pretraining, based on the Huggingface code in `run_language_modelling.py` under the `modelling` dir .
- Code for finetuning AxomiyaBERTa and its phonological version for cross-document coreference resolution on the Translated ECB+ Corpus (Assamese) in `task_finetune/ECB_CDCR` dir.
- Code for finetuning AxomiyaBERTa and its phonological version for NER tasks i.e., AsNER and the WikiNER in  `task_finetune/ASNER` and the `task_finetune/Wiki_NER` dir.
- Code for finetuning AxomiyaBERTa and its phonological version for the Wiki-section Title Prediction task in the `task_finetune/Wiki_section` dir
- Code for finetuning AxomiyaBERTa and its phonological version on the CLoze-QA dataset in the `task_finetune/QA_multiplechoice` dir
- For ECB+, the `scorer` directory inside `task_finetune/ECB_CDCR` dir contain the perl script coreference scorer from `https://github.com/conll/reference-coreference-scorers`.

## Pretraining/Pretrained Model Usage

AxomiyaBERTa is available on HuggingFace at https://huggingface.co/Abhijnan/AxomiyaBERTa. It can be directly used using the from_pretrained method of the Transformers library.





 
