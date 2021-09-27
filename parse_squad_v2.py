import json
from pathlib import Path
import json
import datasets
from random import randint
from nlp import DatasetInfo, BuilderConfig, SplitGenerator, Split, utils
import pandas as pd


logger = datasets.logging.get_logger(__name__)


def read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                for answer in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer["text"]
        start_idx = answer["answer_start"]
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer["answer_end"] = end_idx
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            answer["answer_start"] = start_idx - 1
            answer["answer_end"] = (
                end_idx - 1
            )  # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            answer["answer_start"] = start_idx - 2
            answer["answer_end"] = (
                end_idx - 2
            )  # When the gold label is off by two characters
    return answers


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQuADV2Dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)


class SQuADV2Dataset(datasets.GeneratorBasedBuilder):
    """SQuADV2Dataset: Version 1.0.0"""

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Multitask dataset",
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "answer_start": datasets.Value("int32"),
                    "answer_end": datasets.Value("int32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        contexts, questions, answers = read_squad(filepath)

        answers = add_end_idx(answers, contexts)

        for idx, (context, question, answer) in enumerate(
            zip(contexts, questions, answers)
        ):
            yield idx, {
                "context": context,
                "question": question,
                "answer": answer["text"],
                "answer_start": answer["answer_start"],
                "answer_end": answer["answer_end"],
            }
