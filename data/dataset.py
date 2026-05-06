"""Dataset loading and preprocessing for MATC-Net."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import RobertaTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np


# Dataset metadata
DATASET_INFO = {
    "ag_news": {
        "num_classes": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "imdb": {
        "num_classes": 2,
        "class_names": ["Negative", "Positive"],
    },
    "20newsgroups": {
        "num_classes": 20,
        "class_names": [
            "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
            "comp.sys.mac.hardware", "comp.windows.x", "rec.autos", "rec.motorcycles",
            "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics",
            "sci.med", "sci.space", "talk.politics.misc", "talk.politics.guns",
            "talk.politics.mideast", "talk.religion.misc", "misc.forsale",
            "alt.atheism", "soc.religion.christian",
        ],
    },
    "bbc": {
        "num_classes": 5,
        "class_names": ["business", "entertainment", "politics", "sport", "tech"],
    },
}


class TextClassificationDataset(Dataset):
    """Tokenized text classification dataset."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _load_raw_data(dataset_name: str):
    """Load raw texts and labels for a given dataset."""
    if dataset_name == "ag_news":
        from datasets import load_dataset
        ds = load_dataset("ag_news")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
        return train_texts, train_labels, test_texts, test_labels

    elif dataset_name == "imdb":
        from datasets import load_dataset
        ds = load_dataset("imdb")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
        return train_texts, train_labels, test_texts, test_labels

    elif dataset_name == "20newsgroups":
        train_data = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes")
        )
        test_data = fetch_20newsgroups(
            subset="test", remove=("headers", "footers", "quotes")
        )
        return (
            train_data.data, train_data.target.tolist(),
            test_data.data, test_data.target.tolist(),
        )

    elif dataset_name == "bbc":
        from datasets import load_dataset
        ds = load_dataset("SetFit/bbc-news")
        texts = ds["train"]["text"]
        labels = ds["train"]["label"]
        # Split manually since BBC has no predefined test set
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        return train_texts, train_labels, test_texts, test_labels

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 32,
    max_seq_length: int = 512,
    val_split: float = 0.15,
    num_workers: int = 2,
):
    """Create train, validation, and test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    info = DATASET_INFO[dataset_name]
    num_classes = info["num_classes"]
    class_names = info["class_names"]

    train_texts, train_labels, test_texts, test_labels = _load_raw_data(dataset_name)

    # Split training data into train + validation
    train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=val_split,
        random_state=42,
        stratify=train_labels,
    )

    train_dataset = TextClassificationDataset(
        train_texts_split, train_labels_split, tokenizer, max_seq_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_seq_length
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, max_seq_length
    )

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Classes: {num_classes} — {class_names}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes, class_names
