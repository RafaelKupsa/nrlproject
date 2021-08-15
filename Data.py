import random
from transformers import BertTokenizer, CanineTokenizer
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler


class NextLineData:
    def __init__(self, files):
        self.true_class = []

        for filepath in files:
            with open(filepath, "r", encoding="latin-1") as file:
                prev_line = ""
                for line in file:
                    line = line.strip()
                    if line.startswith("AUTHOR") or line.startswith("RHYME") or len(line) == 0:
                        continue
                    if line.startswith("TITLE"):
                        prev_line = ""
                        continue
                    if prev_line != "":
                        self.true_class.append([prev_line, line])
                    prev_line = line

        self.false_class = []
        for lines in self.true_class:
            rand_line = random.randrange(0, len(self.true_class))

            self.false_class.append([lines[0], self.true_class[rand_line][1]])

    def preprocess(self, model_type, batch_size, num_examples):
        if model_type == "Bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "Canine":
            tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        else:
            raise Exception("model_type must be Bert or Canine")

        input_ids = []
        token_type_ids = []
        attention_mask = []
        for example in self.true_class[:int(num_examples/2)] + self.false_class[:int(num_examples/2)]:
            encoded_data = tokenizer.encode_plus(example[0], text_pair=example[1], padding="max_length", return_tensors="pt")
            input_ids.append(encoded_data["input_ids"])
            token_type_ids.append(encoded_data["token_type_ids"])
            attention_mask.append(encoded_data["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        labels = torch.tensor([0 for _ in self.true_class[:int(num_examples/2)]] + [1 for _ in self.false_class[:int(num_examples/2)]])

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
        train_data, validation_data = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])

        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data),
                                           batch_size=batch_size)

        return train_dataloader, validation_dataloader


class RhymingData:
    def __init__(self, files):
        self.true_class = []
        self.false_class = []

        for filepath in files:
            with open(filepath, "r", encoding="latin-1") as file:
                print("Processing", filepath)
                rhyme_pattern = []
                verse = []
                for line in file:
                    line = line.strip()
                    if line.startswith("AUTHOR") or line.startswith("TITLE") or len(line) == 0:
                        continue
                    if line.startswith("RHYME"):
                        if len(rhyme_pattern) > 0 and rhyme_pattern[-1] == "*":
                            new_rhyme_pattern = rhyme_pattern[:-1]
                            chars = list("bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                            i = 0
                            while len(new_rhyme_pattern) < len(verse):
                                new_rhyme_pattern += [chars[i] for _ in rhyme_pattern[:-1]]
                                i = (i+1) % len(chars)
                            rhyme_pattern = new_rhyme_pattern
                        if len(rhyme_pattern) == len(verse):
                            for i, line_1 in enumerate(verse):
                                for j, line_2 in enumerate(verse):
                                    if i == j:
                                        continue
                                    elif rhyme_pattern[i] == rhyme_pattern[j]:
                                        self.true_class.append([line_1, line_2])
                                        break
                            for i, line_1 in enumerate(verse):
                                for j, line_2 in enumerate(verse):
                                    if i == j:
                                        continue
                                    elif rhyme_pattern[i] != rhyme_pattern[j]:
                                        self.false_class.append([line_1, line_2])
                                        break

                        rhyme_pattern = line.split()[1:]
                        verse = []
                        continue
                    verse.append(line)

    def preprocess(self, model_type, batch_size, num_examples):
        if model_type == "Bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "Canine":
            tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        else:
            raise Exception("model_type must be Bert or Canine")

        input_ids = []
        token_type_ids = []
        attention_mask = []
        for example in self.true_class[:int(num_examples/2)] + self.false_class[:int(num_examples/2)]:
            encoded_data = tokenizer.encode_plus(example[0], text_pair=example[1], padding="max_length", return_tensors="pt")
            input_ids.append(encoded_data["input_ids"])
            token_type_ids.append(encoded_data["token_type_ids"])
            attention_mask.append(encoded_data["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        labels = torch.tensor([0 for _ in self.true_class[:int(num_examples/2)]] + [1 for _ in self.false_class[:int(num_examples/2)]])

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
        train_data, validation_data = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])

        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data),
                                           batch_size=batch_size)

        return train_dataloader, validation_dataloader

