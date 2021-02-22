import os
import torch
import random
import numpy as np
from pytorch_lightning.metrics import Accuracy

from src.data.constants import INDICES_PADDING_VALUE


def _find_nth_word_start_end_indices_in_sentence(sentence, n):
    words = sentence.split(' ')
    start_idx = n
    for word in words[:n]:
        start_idx += len(word)
    return start_idx, start_idx + len(words[n])


def get_word_start_end_in_sentence(row):
    first_word_pos, second_word_pos = [int(idx) for idx in row['word_indices'].split('-')]
    start1, end1 = _find_nth_word_start_end_indices_in_sentence(row['sentence1'], first_word_pos)
    start2, end2 = _find_nth_word_start_end_indices_in_sentence(row['sentence2'], second_word_pos)
    return (start1, end1), (start2, end2)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


def _get_mask(indices, embedding_size):
    mask = (indices != INDICES_PADDING_VALUE)
    mask.unsqueeze_(-1)
    mask = mask.expand(mask.shape[0], mask.shape[1], embedding_size)
    LARGE_VALUE = 2 ** 32
    return torch.where(mask == True, 0, LARGE_VALUE)


def get_tokens_embeddings(batch, indices):
    return _batched_index_select(batch, 1, indices) - _get_mask(indices, batch.shape[2])


def get_max_tokens(dataset):
    tokens = 0
    for item in dataset:
        attention_masks = item[1]
        tokens = max(tokens, attention_masks[0].tolist().index(0), attention_masks[1].tolist().index(0))
    return tokens


def get_max_offset_mappings(dataset):
    mappings = 0
    for item in dataset:
        word_ids_indices = item[2]
        mappings = max(mappings, word_ids_indices[0].tolist().index(INDICES_PADDING_VALUE, 1),
                       word_ids_indices[1].tolist().index(INDICES_PADDING_VALUE, 1))
    return mappings


def get_accuracy(labels, probas, threshold):
    y_pred = (probas > threshold).float()
    acc = Accuracy()
    return acc(y_pred, torch.tensor(labels)).item()
