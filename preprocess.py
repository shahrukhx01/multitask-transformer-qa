import transformers
from transformers import RobertaTokenizerFast


def add_token_positions(encodings, answer_start, answer_end, tokenizer):
    start_positions = []
    end_positions = []
    labels = []
    for i in range(len(answer_start)):
        start_positions.append(encodings.char_to_token(i, answer_start[i]))
        end_positions.append(encodings.char_to_token(i, answer_end[i] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
        labels.append(2)

    encodings.update(
        {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "labels": labels,
        }
    )
    return encodings


def convert_to_features_boolq(
    example_batch, model_name="deepset/roberta-base-squad2", max_length=512
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    questions = list(example_batch["question"])
    passages = list(example_batch["passage"])
    answers = [int(answer) for answer in example_batch["answer"]]

    features = tokenizer(
        questions,
        passages,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = answers
    print(features["labels"])
    return features


def convert_to_features_squad_v2(
    example_batch, model_name="deepset/roberta-base-squad2", max_length=512
):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    encodings = tokenizer(
        example_batch["context"],
        example_batch["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    features = add_token_positions(
        encodings,
        example_batch["answer_start"],
        example_batch["answer_end"],
        tokenizer,
    )
    return features
