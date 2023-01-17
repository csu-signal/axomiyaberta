#!/bin/bash

#SBATCH --job-name="ASNER-NER IndicBERT finetune"	 # job name
#SBATCH --partition=peregrine-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_long			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4         		 # cpu-cores per task
#SBATCH --mem=40G                  		 # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1  # A100 80GB
#SBATCH --time=120:10:00 				 #  wall time

source activate axberta
echo "python3 indic_ner.py"
python3 indic_ner.py