# -*- coding:utf-8 -*-

import typing
from collections import defaultdict
import numpy as np

PAD = '<pad>'
UNK = '[UNK]'
CLS = '[CLS]'
SEP = '[SEP]'
BOS = '<bos>'
START = '<start>'
EOS = '<eos>'
UNDEFINED = 'UNDEFINED'
# 标签BIESO的连接符
TAGGING_LABEL_SEP = '-'


def get_entities(sequence_labels, sep='-', suffix=False):
    """Gets entities from sequence.

    Args:
        sequence_labels: sequence of labels.
        sep
        suffix:

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(sequence_labels)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in sequence_labels):
        sequence_labels = [item for sublist in sequence_labels for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(sequence_labels + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split(sep)[0]
        else:
            tag = chunk[0]
            type_ = chunk.split(sep, 1)[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    new_chunks = []
    for chunk in chunks:
        element = chunk[0]
        start = chunk[1]
        end = chunk[2]
        if len(element.split(sep)) > 0:
            for key in element.split(sep):
                new_chunks.append((key, start, end))
        else:
            new_chunks.append(chunk)
    return new_chunks


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

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
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

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', suffix=False, sep='-'):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, sep, suffix))
    pred_entities = set(get_entities(y_pred, sep, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


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


def precision_score(y_true, y_pred, average='micro', suffix=False, sep='-'):
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, sep, suffix))
    pred_entities = set(get_entities(y_pred, sep, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False, sep='-'):
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, sep, suffix))
    pred_entities = set(get_entities(y_pred, sep, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def performance_measure(y_true, y_pred):
    """
    Compute the performance metrics: TP, FP, FN, TN
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        performance_dict : dict
    Example:
        >>> from seqeval.metrics import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> performance_measure(y_true, y_pred)
        (3, 3, 1, 4)
    """
    performace_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performace_dict['TP'] = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)
                                if ((y_t != 'O') or (y_p != 'O')))
    performace_dict['FP'] = sum(y_t != y_p for y_t, y_p in zip(y_true, y_pred))
    performace_dict['FN'] = sum(((y_t != 'O') and (y_p == 'O'))
                                for y_t, y_p in zip(y_true, y_pred))
    performace_dict['TN'] = sum((y_t == y_p == 'O')
                                for y_t, y_p in zip(y_true, y_pred))

    return performace_dict


def span_classification_report(
        y_true: typing.List[typing.List[str]],
        y_pred: typing.List[typing.List[str]],
        labels: typing.Optional[typing.List[str]] = None,
        digits: int = 3,
        suffix: bool = False,
        sep='-'
):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        labels: 1d array. Labels present in the data can be
                included, for example to calculate a multiclass average ignoring a
                majority negative class, while labels not present in the data will
                result in 'O' in a macro average.
        digits : int. Number of digits for formatting output floating point values.
    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.
    Examples:
        >>> from seqeval.metrics import classification_report
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
    ignore_labels = [
        'O',
        CLS,
        SEP,
        PAD,
        START,
        EOS
    ]
    if labels is not None:
        y_true = [
            [
                'O'
                if i.split(TAGGING_LABEL_SEP)[-1] not in labels else i
                for i in true_
            ]
            for true_ in y_true
        ]
        y_pred = [
            [
                'O'
                if i.split(TAGGING_LABEL_SEP)[-1] not in labels else i
                for i in pred_
            ]
            for pred_ in y_pred
        ]
    else:
        y_true = [
            [
                'O'
                if i.split(TAGGING_LABEL_SEP)[-1] in ignore_labels else i
                for i in true_
            ]
            for true_ in y_true
        ]
        y_pred = [
            [
                'O'
                if i.split(TAGGING_LABEL_SEP)[-1] in ignore_labels else i
                for i in pred_
            ]
            for pred_ in y_pred
        ]

    true_entities = set(get_entities(y_true, sep, suffix))
    pred_entities = set(get_entities(y_pred, sep, suffix))

    # print()
    # print(y_true)
    # print(y_pred)
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'weighted avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        if labels is not None and type_name not in labels:
            continue

        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    total_support = np.sum(s)
    # compute averages
    report += row_fmt.format(
        'micro avg',
        precision_score(y_true, y_pred, suffix=suffix),
        recall_score(y_true, y_pred, suffix=suffix),
        f1_score(y_true, y_pred, suffix=suffix),
        total_support,
        width=width,
        digits=digits
    )
    report += row_fmt.format(
        'macro avg',
        np.average(ps, weights=s),
        np.average(rs, weights=s),
        np.average(f1s, weights=s),
        total_support,
        width=width,
        digits=digits
    )

    report += row_fmt.format(
        last_line_heading,
        np.average(ps, weights=s),
        np.average(rs, weights=s),
        np.average(f1s, weights=s),
        total_support,
        width=width,
        digits=digits
    )

    return report


def parse_metrics_from_report(report: str):
    precision, recall, f1 = None, None, None
    for line in report.split('\n'):
        if 'micro avg' in line:
            *_, precision, recall, f1, _ = line.split()
            return {
                'precision': float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        elif 'avg' in line and 'total' in line:
            *_, precision, recall, f1, _ = line.split()

    if None in [precision, recall, f1]:
        raise Exception(f'input report not valid: {report}')
    return {
        'precision': float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


if __name__ == '__main__':
    y_true = [['[CLS]', 'B-O', 'O', 'O', 'B-MISC-A', 'I-MISC-A', 'I-MISC-A', '[SEP]'], ['B-PER', 'I-PER', 'O']]
    y_pred = [['[CLS]', 'O', 'O', 'B-MISC-A', 'I-MISC-A', 'I-MISC-A', 'I-MISC-A', '[SEP]'], ['B-PER', 'I-PER', 'O']]
    print(f1_score(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    print('span:\n', span_classification_report(y_true, y_pred))
