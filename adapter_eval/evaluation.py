# This script will go through all adapters in trained_adapters and return their score on Afrikaans NER
# We trained different language family adapters and configurations on CC100
# now it's time to evaluate down-stream performance
import gc
import json


from transformers import TrainingArguments, AutoTokenizer
from adapters import AdapterTrainer, init
from adapters.composition import Stack

from datasets import load_dataset
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification

import torch
import numpy as np
import evaluate

metric = evaluate.load("seqeval")


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
dataset_af = load_dataset("wikiann", "af", trust_remote_code=True)
# If True, all tokens of a word will be labeled, otherwise only the first token
label_all_tokens = True


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


# Batch encoding function for NER
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# Load and preprocess WikiANN (English)


tokenized_datasets = dataset_af.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset_af["train"].column_names,
)

# dataset_en is now ready to be used with adapters for cross-lingual transfer

ner_feature = dataset_af["train"].features["ner_tags"]

label_names = ner_feature.feature.names
words = dataset_af["train"][0]["tokens"]
labels = dataset_af["train"][0]["ner_tags"]
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[lab] for lab in label if lab != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    id2label=id2label,
    label2id=label2id,
)
init(model)
# we load in the adapters
model.load_adapter("../trained_adapters/ner")
model.load_adapter("../trained_adapters/family/en-af/mlm", load_as="en-af")
# model.active_adapters = Stack("en-af", "ner")
model.active_adapters = "ner"

eval_trainer = AdapterTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./eval_output",
        remove_unused_columns=False,
    ),
    data_collator=data_collator,
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)
ev = eval_trainer.evaluate()
print(ev)

eval_dict = {"ner": ev}


def ner_eval(combi, eval_dict=None, mono=False):
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        id2label=id2label,
        label2id=label2id,
    )
    init(model)
    # we load in the adapters
    model.load_adapter("../trained_adapters/ner")
    if mono:
        model.load_adapter(f"../trained_adapters/mono/{combi}", load_as=combi)
    else:
        model.load_adapter(f"../trained_adapters/family/{combi}/mlm", load_as=combi)
    model.active_adapters = Stack(combi, "ner")
    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_output",
            remove_unused_columns=False,
        ),
        data_collator=data_collator,
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    ev = eval_trainer.evaluate()
    print(ev)
    if eval_dict:
        eval_dict[combi] = ev
    # we empty the cache and model
    model.cpu()
    del model
    del eval_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # We check how much memory we are using


to_eval = ["en-af", "en-de-af", "en-de-nl-af", "en-tr-af", "yo-eu-tr-af"]

for combi in to_eval:
    print(combi)
    ner_eval(combi, eval_dict=eval_dict)

monos = ["en", "de", "nl"]
for comb in monos:
    print(comb)
    ner_eval(comb, eval_dict=eval_dict, mono=True)


print(eval_dict)
# save the eval_dict to a json file
with open("eval_dict.json", "w") as f:
    json.dump(eval_dict, f)
