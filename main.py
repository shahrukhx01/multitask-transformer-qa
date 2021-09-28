import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm


from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import BertForSequenceClassification
from preprocess import convert_to_features_boolq, convert_to_features_squad_v2
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn
from checkpoint_model import save_model
from pathlib import Path


logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    dataset_dict = {
        "squad_v2": load_dataset(
            "parse_squad_v2.py",
            data_files={
                "train": "squad/train-v2.0.json",
                "validation": "squad/dev-v2.0.json",
            },
        ),
        "boolq": load_dataset("boolq"),
    }

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print("train", dataset_dict[task_name]["train"][0])
        print("validation", dataset_dict[task_name]["validation"][0])
        print()

    multitask_model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        task_labels_map={"squad_v2": 2, "boolq": 2},
    )

    print(multitask_model.bert.embeddings.word_embeddings.weight.data_ptr())

    convert_func_dict = {
        "boolq": convert_to_features_boolq,
        "squad_v2": convert_to_features_squad_v2,
    }

    columns_dict = {
        "squad_v2": [
            "input_ids",
            "attention_mask",
            "answer_start",
            "answer_end",
            "answer",
            "start_positions",
            "end_positions",
        ],
        "boolq": ["input_ids", "attention_mask", "labels"],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }
    """
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=args.num_train_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=args.per_device_train_batch_size,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()

    ## evaluate on given tasks
    multitask_eval_fn(multitask_model, args.model_name_or_path, dataset_dict)

    ## save model for later use
    save_model(args.model_name_or_path, multitask_model)"""


if __name__ == "__main__":
    main()
