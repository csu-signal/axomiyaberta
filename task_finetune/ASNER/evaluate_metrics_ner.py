#Author: Abhijnan Nath, CSU Signal Lab. Extended from the seqeval source code (https://huggingface.co/spaces/evaluate-metric/seqeval) to 
#suit Assamese NER dataset framework i.e., non-(IOB1 IOB2 IOE1 IOE2 IOBES) and added code for partial token overlap evaluation metrics. 

import warnings
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from seqeval.reporters import DictReporter, StringReporter
from seqeval.scheme import Entities, Token, auto_detect

from collections import defaultdict

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]

def precision_recall_fscore_support_asner(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False, overlap ='full') :
    """Compute precision, recall, F-measure and support for each class.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.

        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        suffix : bool, False by default.

    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]

        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]

        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]

        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.

    Examples:
        >>> from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        (0.5, 0.5, 0.5, 2)

        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:

        >>> precision_recall_fscore_support(y_true, y_pred, average=None)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))

    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """
    
    def extract_tp_actual_correct(y_true, y_pred, suffix, overlap =None,  *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, start, end in get_entities_asner(y_true, suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities_asner(y_pred, suffix):
            entities_pred[type_name].add((start, end))
        
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
       
        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
           
        
        overlap ='full'  #  Set either full overlap for evaluating NER or partial token overlap
        for type_name in target_names:
            
            
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            entities_true_list = list(entities_true_type)
            entities_pred_list= list(entities_pred_type)
           
            if overlap =='full':
                print(overlap)
                
                tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
              
                pred_sum = np.append(pred_sum, len(entities_pred_type))
                true_sum = np.append(true_sum, len(entities_true_type))
           
            else:
            
                overlap_count =0
                
                tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
 
                for i, (j, k) in enumerate(zip(sorted(entities_true_list), sorted( entities_pred_list))):
      
                
                    if i<len(entities_pred_type)-1:
                    
                        if (j[1]==k[1] and k[0]!=j[0]) or (j[0]==k[0] and j[1]!=k[1]) or sorted( entities_pred_list)[i+1][0] ==sorted( entities_true_list)[i][1] :
                            overlap_count+=1
                            
                if overlap_count>=1:
                    
                    tp_sum[-1] = tp_sum[-1]+ overlap_count
                 
                else:
                    tp_sum =tp_sum

                pred_sum = np.append(pred_sum, len(entities_pred_type))
                true_sum = np.append(true_sum, len(entities_true_type))
    

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct 
    )

    return precision, recall, f_score, true_sum


import warnings
def get_entities_asner(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
        
        
    """
 
    if any(isinstance(s, list) for s in seq):
        
        seq = [item for sublist in seq for item in sublist ]
    
    prev_tag = 'S'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['S']):
 
        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk
            
            type_ = chunk[0:].split('-', maxsplit=1)[-1] or '_'
 
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
            #print(" end chunkcs", chunks)
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            #print("start chunk")
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = False
    

    if prev_tag != 'S' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'S' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start



def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score

 

def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = 'sample', 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0
    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s. Use `zero_division` parameter to control'
           ' this behavior.'.format(msg_start, modifier, axis0))
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


def unique_labels(y_true: List[List[str]], y_pred: List[List[str]],
                  scheme: Type[Token], suffix: bool = False) -> List[str]:
    sequences_true = Entities(y_true, scheme, suffix)
    sequences_pred = Entities(y_pred, scheme, suffix)
    unique_tags = sequences_true.unique_tags | sequences_pred.unique_tags
    return sorted(unique_tags)


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """Check that all arrays have consistent first and second dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if not is_list == {list}:
        raise TypeError('Found input variables without list of list.')

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)


def _precision_recall_fscore_support(y_true: List[List[str]],
                                     y_pred: List[List[str]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     scheme: Optional[Type[Token]] = None,
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None) -> SCORES:
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    check_consistent_length(y_true, y_pred)

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, suffix, scheme)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)

    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    scheme: Optional[Type[Token]] = None,
                                    suffix: bool = False,
                                    **kwargs) -> SCORES:
    """Compute precision, recall, F-measure and support for each class.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.

        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]

        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]

        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]

        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.

    Examples:
        >>> from seqeval.metrics.v1 import precision_recall_fscore_support
        >>> from seqeval.scheme import IOB2
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)

        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:

        >>> precision_recall_fscore_support(y_true, y_pred, average=None, scheme=IOB2)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))

    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """
    def extract_tp_actual_correct(y_true, y_pred, suffix, scheme):
        # If this function is called from classification_report,
        # try to reuse entities to optimize the function.
        entities_true = kwargs.get('entities_true') or Entities(y_true, scheme, suffix)
        entities_pred = kwargs.get('entities_pred') or Entities(y_pred, scheme, suffix)
        target_names = sorted(entities_true.unique_tags | entities_pred.unique_tags)

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.filter(type_name)
            entities_pred_type = entities_pred.filter(type_name)
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
            print("tpsum", tp_sum)
            print("predsum", pred_sum)
            print("true sum ", true_sum)

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=scheme,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def classification_report(y_true: List[List[str]],
                          y_pred: List[List[str]],
                          *,
                          sample_weight: Optional[List[int]] = None,
                          digits: int = 2,
                          output_dict: bool = False,
                          zero_division: str = 'warn',
                          suffix: bool = False,
                          scheme: Type[Token] = None) -> Union[str, dict]:
    """Build a text report showing the main tagging metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a classifier.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        digits : int. Number of digits for formatting output floating point values.

        output_dict : bool(default=False). If True, return output as dict else str.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        report : string/dict. Summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics.v1 import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
       weighted avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    check_consistent_length(y_true, y_pred)

    if scheme is None or not issubclass(scheme, Token):
        scheme = auto_detect(y_true, suffix)

    entities_true = Entities(y_true, scheme, suffix)
    entities_pred = Entities(y_pred, scheme, suffix)
    target_names = sorted(entities_true.unique_tags | entities_pred.unique_tags)

    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, target_names))
        avg_width = len('weighted avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    # compute per-class scores.
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=scheme,
        suffix=suffix,
        entities_true=entities_true,
        entities_pred=entities_pred
    )
    for row in zip(target_names, p, r, f1, s):
        reporter.write(*row)
    reporter.write_blank()

    # compute average scores.
    average_options = ('micro', 'macro', 'weighted')
    for average in average_options:
        avg_p, avg_r, avg_f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
            scheme=scheme,
            suffix=suffix,
            entities_true=entities_true,
            entities_pred=entities_pred
        )
        reporter.write('{} avg'.format(average), avg_p, avg_r, avg_f1, support)
    reporter.write_blank()

    return reporter.report()

