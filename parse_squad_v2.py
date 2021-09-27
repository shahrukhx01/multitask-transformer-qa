import json
from pathlib import Path


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


def get_squad_v2():
    train_contexts, train_questions, train_answers = read_squad("squad/train-v2.0.json")
    val_contexts, val_questions, val_answers = read_squad("squad/dev-v2.0.json")

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    squad_dict = {}
    train_examples, val_examples = [], []
    for train_context, train_question, train_answer in zip(
        train_contexts, train_questions, train_answers
    ):
        train_examples.append(
            {
                "question": train_question,
                "context": train_context,
                "answer": train_answer,
            }
        )

    for val_context, val_question, val_answer in zip(
        val_contexts, val_questions, val_answers
    ):
        val_examples.append(
            {
                "question": val_question,
                "context": val_context,
                "answer": val_answer,
            }
        )

    squad_dict["train"] = train_examples

    squad_dict["validation"] = val_examples
    return squad_dict
