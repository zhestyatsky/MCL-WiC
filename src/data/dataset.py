import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from transformers import BertTokenizerFast, XLMRobertaTokenizerFast

from src.data.processing import get_ids, get_sentences, get_word_ranges, get_labels
from src.data.reading import get_train_val_test_df
from src.data.constants import INDICES_PADDING_VALUE, INDICES_PADDING_LEN, MAX_TOKENS, BATCH_SIZE
from src.training.config import get_config


class BertDataset(Dataset):
    def __init__(self, model_path, ids, sentences, word_ranges, max_tokens, labels=None):
        if model_path == "bert-base-cased" or model_path == "bert-large-cased":
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        elif model_path == "xlm-roberta-large" or model_path == "xlm-roberta-base":
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)
        else:
            raise RuntimeError("Specify correct embeddings: " + model_path)

        self.ids = ids
        self.sentences = sentences
        self.word_ranges = word_ranges
        self.labels = labels
        self.max_tokens = max_tokens

    def _tokenize(self, sentence):
        return self.tokenizer(sentence,
                              add_special_tokens=True,
                              max_length=self.max_tokens,
                              padding="max_length",
                              truncation=True,
                              return_offsets_mapping=True)

    @staticmethod
    def _get_input_ids_indices_for_word(offset_mapping, word_start, word_end):
        indices = []
        for idx, (start, end) in enumerate(offset_mapping):
            if start != end and word_start <= start and end <= word_end:
                indices.append(idx)
            elif word_start < start:
                break

        indices.extend([INDICES_PADDING_VALUE for i in range(INDICES_PADDING_LEN - len(indices))])
        return torch.tensor(indices)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        first_sentence, second_sentence = self.sentences[index]
        (first_word_start, first_word_end), (second_word_start, second_word_end) = self.word_ranges[index]

        first_input = self._tokenize(first_sentence)
        second_input = self._tokenize(second_sentence)

        input_ids = (torch.tensor(first_input["input_ids"]), torch.tensor(second_input["input_ids"]))
        attention_masks = (torch.tensor(first_input["attention_mask"]), torch.tensor(second_input["attention_mask"]))

        first_word_ids_indices = self._get_input_ids_indices_for_word(first_input["offset_mapping"], first_word_start,
                                                                      first_word_end)
        second_word_ids_indices = self._get_input_ids_indices_for_word(second_input["offset_mapping"],
                                                                       second_word_start, second_word_end)

        word_ids_indices = (first_word_ids_indices, second_word_ids_indices)

        label = self.labels[index] if self.labels is not None else 2

        return input_ids, attention_masks, word_ids_indices, torch.tensor(label, dtype=torch.float)


def get_loader(model_path, df, is_train=False):
    labels = get_labels(df)
    data = BertDataset(model_path, get_ids(df), get_sentences(df), get_word_ranges(df), MAX_TOKENS, labels=labels)
    sampler = RandomSampler(data) if is_train else None
    loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=sampler)
    return loader


def get_train_val_test_loaders(model_description, on_colab=True):
    model_config = get_config(model_description)
    model_path = model_config["embeddings"]
    train_df, val_df, test_df = get_train_val_test_df(use_default_datasets=model_config["use_default_datasets"],
                                                      only_wic=model_config["only_wic"], on_colab=on_colab)
    train_loader = get_loader(model_path, train_df, is_train=True)
    val_loader = get_loader(model_path, val_df)
    test_loader = get_loader(model_path, test_df)
    return train_loader, val_loader, test_loader
