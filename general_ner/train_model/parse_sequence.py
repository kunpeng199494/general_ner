# -*- coding:utf-8 -*-

import argparse
import typing
import hao

LOGGER = hao.logs.get_logger(__name__)


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


def get_total_entities(
        sequence_labels: typing.List[str],
        suffix=False,
        sep='-',
        default_label="O"
):
    entities = get_entities(sequence_labels, sep, suffix)

    pointer = 0
    whole_labels = []
    for label, st, ed in entities:
        if st > pointer:
            whole_labels.append((default_label, pointer, st - 1))

        whole_labels.append((label, st, ed))
        pointer = ed + 1

    if pointer < len(sequence_labels):
        whole_labels.append((default_label, pointer, len(sequence_labels)))

    return whole_labels


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    seq = [['B-PER-A-C', 'I-PER-A-C', 'B-O', 'E-O', 'B-LOC']]
    result = get_total_entities(seq)

    print(result)
