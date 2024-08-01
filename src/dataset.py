import typing as tp
import torch
import trl
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RewardDatasetItem(tp.TypedDict):
    input_ids_chosen: list[int]
    attention_mask_chosen: list[int]
    input_ids_rejected: list[int]
    attention_mask_rejected: list[int]


class DatasetItem(tp.TypedDict):
    text: str
    query: str
    input_ids: torch.Tensor


class IMDBRewardDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, accepted_label: int, split: str = 'train'):
        super().__init__()

        imdb_dataset = datasets.load_dataset('stanfordnlp/imdb', split=split)
        self.tokenizer = tokenizer
        self.chosen_texts = [row['text'] for row in imdb_dataset if row['label'] == accepted_label]
        self.rejected_texts = [row['text'] for row in imdb_dataset if row['label'] != accepted_label]

        self.n_chosen = len(self.chosen_texts)
        self.n_rejected = len(self.rejected_texts)

    def __len__(self):
        return self.n_chosen * self.n_rejected

    def __getitem__(self, index: int) -> RewardDatasetItem:
        chosen = self.tokenizer(self.chosen_texts[index // self.n_rejected], truncation=True)
        rejected = self.tokenizer(self.rejected_texts[index % self.n_rejected], truncation=True)

        return dict(input_ids_chosen=chosen['input_ids'], attention_mask_chosen=chosen['attention_mask'],
                    input_ids_rejected=rejected['input_ids'], attention_mask_rejected=rejected['attention_mask'])


def select_query_and_tokenize(sample: DatasetItem, tokenizer: PreTrainedTokenizer, length_sampler: trl.core.LengthSampler):
    query_ids = tokenizer.encode(sample['text'])[:length_sampler()]
    sample["query"] = tokenizer.decode(query_ids)
    sample["input_ids"] = query_ids
    return sample


def build_imdb_dataset(tokenizer: PreTrainedTokenizer, min_text_length: int = 200, min_tokens: int = 5, max_tokens: int = 15) -> Dataset[DatasetItem]:
    imdb_dataset = datasets.load_dataset('stanfordnlp/imdb')
    length_sampler = trl.core.LengthSampler(min_tokens, max_tokens)

    imdb_dataset = imdb_dataset.filter(lambda row: len(row['text']) > min_text_length, batched=False)
    # Need to have label column to make compute_metrics work
    # imdb_dataset = imdb_dataset.remove_columns(['label'])
    imdb_dataset = imdb_dataset.map(lambda sample: select_query_and_tokenize(sample, tokenizer, length_sampler), batched=False)
    imdb_dataset.set_format(type='torch')

    return imdb_dataset
