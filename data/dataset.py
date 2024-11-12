from torch.utils.data import Dataset
from typing import Dict, List
from transformers import AutoTokenizer
import pandas as pd
import torch

class PromptuningDataset(Dataset):
    def __init__(self, csv_file:str, tokenizer: AutoTokenizer, input_max_length: int, text_column: str, label_column: str) -> None:
        self.data = pd.read_csv(csv_file)
        classes = list(self.data[label_column].unique())
        self.data[label_column] = self.data[label_column].apply(lambda x:[classes.index(x)])
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.N = len(self.data)
        self.text_column = text_column
        self.label_column = label_column

        self.processed_datasets = self.preprocess_data(self.data)
        self.processed_datasets = [dict(zip(self.processed_datasets, values)) for values in zip(*self.processed_datasets.values())]
        
    def preprocess_data(self, data):
        inputs = [f"{self.text_column} : {x} Label : " for x in data[self.text_column]]
        targets = [str(x) for x in data[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)

        for i in range(self.N):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(self.N):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.input_max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.input_max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.input_max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.input_max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.input_max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.input_max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __len__(self):
        return len(self.processed_datasets)
    
    def __getitem__(self, index) -> Dict:
        if torch.is_tensor(index):
            index = index.tolist()

        return self.processed_datasets[index]
        
