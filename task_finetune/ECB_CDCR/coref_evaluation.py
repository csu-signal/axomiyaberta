import os
import sys
sys.path.insert(0, os.getcwd())
sys.argv = ['']
import pickle
import argparse
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from eval_coref import *
#from evaluations.eval_ramfis import *
import numpy as np
# from coreference.incremental_clustering import incremental_clustering
from sklearn.cluster import AgglomerativeClustering
import gc
gc.collect()
import torch

#parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
import os.path
from sklearn.model_selection import train_test_split
import pyhocon
#from coreference.models import LongFormerCrossEncoder, LongFormerCosAlign
import random
from tqdm.autonotebook import tqdm
import csv
import panphon
import panphon.distance
import editdistance # levenshtein
import epitran


parent_path = "../../"
working_folder = parent_path + "/As_Indic_data/ecb/"

 
tn_triv_test_path = working_folder + '/lemma_balanced_tn_fn_test.tsv'
tp_triv_test_path = working_folder + '/lemma_balanced_tp_fp_test.tsv'
tp_triv_dev_path = working_folder + '/lemma_balanced_tp_fp_dev.tsv'
tn_triv_dev_path = working_folder + '/lemma_balanced_tn_fn_dev.tsv'
 
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
    
def get_mention_pair_similarity_lemma(mention_pairs, mention_map, relations, working_folder):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    similarities = []

    within_doc_similarities = []

    doc_sent_map = pickle.load(open(working_folder + '/doc_sent_map.pkl', 'rb'))
    #doc_sims = pickle.load(open(working_folder + '/doc_sims_path.pkl', 'rb'))
    doc_ids = []

    for doc_id, _ in list(doc_sent_map.items()):
        doc_ids.append(doc_id)

    doc2id = {doc: i for i, doc in enumerate(doc_ids)}

    # generate similarity using the mention text
    for pair in mention_pairs:
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['mention_text'].lower()
        men_text2 = men_map2['mention_text'].lower()

        def jc(arr1, arr2):
            return len(set.intersection(arr1, arr2))/len(set.union(arr1, arr2))

        sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        #doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(men_map1['ssameselemma'] in men_text2 or men_map2['lemma'] in men_text1)

        same_doc = float(men_map1['doc_id'] == men_map2['doc_id'])

        # similarities.append(sent_sim + doc_sim + lemma_sim)
        similarities.append((lemma_sim + 0.3*sent_sim)/2)
        # similarities.append((lemma_sim + 0.3*sent_sim)/2)
        within_doc_similarities.append(same_doc)


        # doc_plus_sent = 0. + doc_sim + sent_sim
        # if men_map1['lemma'] in men_text2 or men_map2['lemma'] in men_text1:
        #     # similarities.append(jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens'])))
        #     similarities.append(1. + doc_plus_sent)
        #
        #     if men_map1['doc_id'] == men_map2['doc_id']:
        #         within_doc_similarities.append(1 + sent_sim)
        #     else:
        #         within_doc_similarities.append(doc_sim)
        #
        # else:
        #     similarities.append(doc_sim + sent_sim)
        #     if men_map1['doc_id'] == men_map2['doc_id']:
        #         within_doc_similarities.append(sent_sim)
        #     else:
        #         within_doc_similarities.append(doc_sim)

    combined_sim = np.array(similarities) + np.array(within_doc_similarities)

    return np.array(similarities) 
def get_mention_pair_similarity_lemma_assamese(mention_pairs, mention_map, relations, working_folder):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    
    dst = panphon.distance.Distance()


    epi = epitran.Epitran('eng-Latn')

    
    similarities = []

    within_doc_similarities = []

    doc_sent_map = pickle.load(open(working_folder + '/doc_sent_map.pkl', 'rb'))
    #doc_sims = pickle.load(open(working_folder + '/doc_sims_path.pkl', 'rb'))
    doc_ids = []

    for doc_id, _ in list(doc_sent_map.items()):
        doc_ids.append(doc_id)

    doc2id = {doc: i for i, doc in enumerate(doc_ids)}

    # generate similarity using the mention text
    for pair in mention_pairs:
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['assamese_lemma'].lower()
        
        men_text2 = men_map2['assamese_lemma'].lower()
        print(men_text1,men_text2)

        def jc(arr1, arr2):
            return len(set.intersection(arr1, arr2))/len(set.union(arr1, arr2))

        #sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        #doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        #lemma_sim = float(men_map1['lemma'] in men_text2 or men_map2['lemma'] in men_text1)
        #lemma_sim = float(men_map1['assamese_lemma'] in men_text2 or men_map2['assamese_lemma'] in men_text1)

        lemma_sim = float(men_map1['assamese_lemma']==men_map2['assamese_lemma'])

        same_doc = float(men_map1['doc_id'] == men_map2['doc_id'])
        within_doc_similarities.append(same_doc)

        similarities.append(lemma_sim)
        
        combined_sim = np.array(similarities) + np.array(within_doc_similarities)
        return np.array(similarities)

def get_mention_pair_similarity_lemma_simple(mention_pairs, mention_map, relations, working_folder):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    similarities = []

    # generate similarity using the mention text
    for pair in mention_pairs:
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['assamese_lemma']
        men_text2 = men_map2['assamese_lemma']
        similarities.append(int(men_text1 == men_text2))

    return np.array(similarities)

def cluster_cc(affinity_matrix, threshold=0.8):
    """
    Find connected components using the affinity matrix and threshold -> adjacency matrix
    Parameters
    ----------
    affinity_matrix: np.array
    threshold: float

    Returns
    -------
    list, np.array
    """
    adjacency_matrix = csr_matrix(affinity_matrix > threshold)
    clusters, labels = connected_components(adjacency_matrix, return_labels=True, directed=False)
    return clusters, labels

def cluster_agglo(affinity_matrix, threshold=-1):
    """
    Agglomerative clustering based on the affinity matrix
    :param affinity_matrix: np.array
        The similarity matrix. Need to convert into distance matrix for agglo clustering algo
    :param threshold: float
        Linkage threshold
    :return: list, np.array
        The labels of the nodes
    """
    clustering_ = AgglomerativeClustering(n_clusters=None,
                                          affinity='precomputed',
                                          linkage='average',
                                          distance_threshold=threshold)
    # convert affinity into distance
    distance_matrix = 1 - np.array(affinity_matrix)
    # fit predict
    labels = clustering_.fit_predict(distance_matrix)
    # get clusters
    clusters = defaultdict(list)
    for ind, label in enumerate(labels):
        clusters[label].append(ind)
    return list(clusters.values()), labels
    
def run_coreference(ann_dir, working_folder, men_type='evt', split='test'):
    """

    Parameters
    ----------
    ann_dir
    working_folder
    men_type
    split

    Returns
    -------

    """
    # read annotations
    
    parent_path = "../../"
 
    
    working_folder = parent_path + "/As_Indic_data/ecb/"
    
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == men_type and
                                                    val['split'] == split}
    
    
    
    train_mention_map_path = working_folder + '/train_mention_map_ecb.pkl'
    dev_mention_map_path = working_folder + '/dev_mention_map_ecb.pkl'
    test_mention_map_path = working_folder + '/test_mention_map_ecb.pkl'
    


        
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

                            
    curr_mention_map  = test_mention_map
    
    
    
    
    
    # do some filtering:
#     curr_mention_map_new = {}
#     for key, mention in curr_mention_map.items():
#         mention_text = mention['mention_text']
#         if len(mention_text.strip()) > 2 and len(mention_text.split()) < 4:
#             curr_mention_map_new[key] = mention

    simulation = False
    
    
    #creating the mention pairs 
    
    all_mention_map =ecb_mention_map
                            
    curr_mentions = sorted(list(curr_mention_map.keys()), key=lambda x: curr_mention_map[x]['m_id'])
    print("curr mentions", len(curr_mentions))
    curr_men_to_ind = {eve: i for i, eve in enumerate(curr_mentions)}
    

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, all_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = working_folder + f'/{men_type}_gold.keyfile'
    generate_key_file(curr_gold_cluster_map, men_type, working_folder, gold_key_file)

    # group mentions by topic
    topic_mention_dict = defaultdict(list)
                            
    
    for men_id, coref_map in test_mention_map.items():
        topic = coref_map['topic']
        topic_mention_dict[topic].append(men_id)

    # generate mention-pairs
    mention_pairs = []
    for mentions in topic_mention_dict.values():
        list_mentions = list(mentions)
        for i in range(len(list_mentions)):
            for j in range(i + 1):
                if i != j:
                    mention_pairs.append((list_mentions[i], list_mentions[j]))
    
    
    print("mentions pairs",len(mention_pairs))  
    
######### GET TEST and DEV Scores from running the pairwise_scorer.py script and load them here! ##########        

        #dev set 
#     scores_folder = working_folder + f'/dev_scores/'
#     scores = pickle.load(open(scores_folder + '/dev_prediction_scores', 'rb'))
#     pairs = pickle.load(open(scores_folder + '/dev_pairs', 'rb'))
#     labels = pickle.load(open(scores_folder + '/dev_labels', 'rb'))
    
#         #test set



    scores_folder = working_folder + f'/test_scores/'  #axbert
    #scores_folder = working_folder + f'/muril_test_scores/'  # muril 
    scores_folder = working_folder + f'/axbert_test_scores_pan_upsample/'  # Indic bert
    
    
    scores = pickle.load(open(scores_folder + '/test_prediction_scores', 'rb'))
    sig = pickle.load(open(scores_folder + '/test_prediction_sigmoids', 'rb'))
    
    pairs = pickle.load(open(scores_folder + '/test_pairs', 'rb'))
    labels = pickle.load(open(scores_folder + '/test_labels', 'rb'))
    
    
    scores = scores.detach().cpu().numpy()
    sig = sig.detach().cpu().numpy()
    similarities_coref = np.array(scores.squeeze())
    sig  = np.array(sig.squeeze() )
    
    similarities_coref = similarities_coref>7
    

    ground_truth_nt = np.array([ecb_mention_map[m1]['gold_cluster'] == ecb_mention_map[m2]['gold_cluster'] for m1, m2 in non_triv_mention_pairs]) 
    ground_truth_t = np.array([ecb_mention_map[m1]['gold_cluster'] == ecb_mention_map[m2]['gold_cluster'] for m1, m2 in triv_mention_pairs]) 
 
    
    # Comment out similarities_lemma when not using the lemma-based string similarity heuristic!
    
    #similarities_lemma = get_mention_pair_similarity_lemma_assamese(mention_pairs , curr_mention_map, relations, working_folder)
    similarities_lemma = get_mention_pair_similarity_lemma_simple(mention_pairs, test_mention_map, relations, working_folder)
    #similarities_long = np.array(ab_scores)
    similarities_lemma = similarities_lemma>0.2


    
#     print('Triv all positives:', similarities_lemma.sum())
#     print('Ground truth triv all positives:', ground_truth_t.sum())
#     print('Triv true positives:', np.logical_and(similarities_lemma, ground_truth_t).sum())
#     print('Triv false positives:', np.logical_and(similarities_lemma, np.logical_not(ground_truth_t)).sum())
#     print('Triv true negatives:', np.logical_and(np.logical_not(similarities_lemma), np.logical_not(ground_truth_t)).sum())
#     print('Triv false negatives:', np.logical_and(np.logical_not(similarities_lemma), ground_truth_t).sum())
    
    
#     print('NON Triv all positives:', similarities_long.sum())
#     print('Ground truth all positives:', ground_truth_nt.sum())
#     print('NON Triv true positives:', np.logical_and(similarities_long, ground_truth_nt).sum())
#     print('NON Triv false positives:', np.logical_and(similarities_long, np.logical_not(ground_truth_nt)).sum())
#     print('NON Triv true negatives:', np.logical_and(np.logical_not(similarities_long), np.logical_not(ground_truth_nt)).sum())
#     print('NON Triv false negatives:', np.logical_and(np.logical_not(similarities_long), ground_truth_nt).sum())
    
    


    all_mention_pairs = pairs
 
    mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in all_mention_pairs]
       
    #mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in test_pairs]
    rows, cols = zip(*mention_ind_pairs)
    print(len(rows), len(cols))

    # create similarity matrix from the similarities
    n = len(curr_mentions)
    
    #change current mentions for triviality pairs 
    #n = len(test_pairs)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] =similarities_coref
    #clusters, labels = cluster_cc(similarity_matrix, threshold=0.5)

    # clustering algorithm and mention cluster map
#     if cluster_algo == 'cc':
    #clusters, labels = cluster_cc(similarity_matrix)
    clusters, labels = cluster_agglo(similarity_matrix)
    print(np.sum(similarity_matrix))

        
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)


def _generate_simulation_results_plot(men_type='evt', split='dev'):
   
    working_folder = "../parsing/ecb"
    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == men_type and
                        val['split'] == split}

    simulation = True

    if not simulation:
        coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                    cluster_algo='inc', threshold=0.4, simulation=simulation, top_n=3)
    else:
        top_ns = [3, 5, 10, 20]

        simulation_metrics_n = []

        for n in top_ns:
            simulation_metrics = coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                                             cluster_algo='inc', threshold=0.1, simulation=True, top_n=n)
            simulation_metrics_n.append(simulation_metrics)

        comparisons_plot(zip(top_ns, simulation_metrics_n))


def comparisons_plot(results):
    ns, sim_metrics = zip(*results)
    recalls, precisions, comparisons = zip(*sim_metrics)

    print(sim_metrics)

    import matplotlib.pyplot as plt

    plt.plot(comparisons, recalls, marker='x')
    plt.show()
#_generate_simulation_results_plot()

parent_path = "../../"
working_folder = parent_path + "/As_Indic_data/ecb/"
ann_dir = #path to the ECB Plus Dataset available online at https://github.com/cltl/ecbPlus/tree/master/ECB%2B_LREC2014
    
similarities_lemma  = run_coreference(ann_dir, working_folder, men_type='evt', split='test')