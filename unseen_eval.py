import re
from collections import OrderedDict
import torch
import evaluate
import collections
from huggingface_hub import HfApi
from qq import LanguageData, TagType
from urielplus import urielplus
from datasets import load_dataset, get_dataset_config_names, Dataset, Split
from transformers import (
    TrainingArguments,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import EvalPrediction
from tqdm import tqdm
import os

metrics = {"ner": evaluate.load("seqeval"), "qa": evaluate.load("squad"), "sib": evaluate.load("accuracy")}

api = HfApi()

ld = LanguageData.from_db()


u = urielplus.URIELPlus()
u.set_cache(True)
try:
    u.integrate_grambank()
except SystemExit:
    print("already using GramBank")
try:
    u.set_glottocodes()
except SystemExit:
    print("already using Glottocodes")

task2ds = {
    "ner": "unimelb-nlp/wikiann",
    "pos": "universal_dependencies",
    "copa": "xcopa",
    "qa": "google/xquad",
    "sib": "Davlan/sib200",
}


def get_eval_languages(task):
    if task in ["ner", "copa"]:
        eval_languages = get_dataset_config_names(task2ds[task])
        return {lan: lan for lan in eval_languages if len(lan) <= 3}
    elif task in ["pos", "qa"]:
        ud_datasets = get_dataset_config_names(task2ds[task])
        eval_languages = {}
        for ds in ud_datasets:
            lang = ds.split("_")[0] if task == "pos" else ds.split(".")[1]
            try:
                ld.get(lang, tag_type=TagType.BCP_47_CODE)
                eval_languages[lang] = ds
            except KeyError:
                # print(f"Language {lang} not in database, skipping")
                continue
        return eval_languages
    elif task == "sib":
        langs_scripts = get_dataset_config_names(task2ds[task])
        langs_scripts = [lang for lang in langs_scripts if "Nkoo" not in lang]
        eval_languages = {}
        for ds in langs_scripts:
            lang, script = ds.split("_")
            try:
                name = ld.get(lang, tag_type=TagType.ISO_639_3_CODE).bcp_47_code
                eval_languages[(name, script)] = ds
            except KeyError:
                # print(f"Language {lang} not in database, skipping)
                continue
        # print(eval_languages)
        return eval_languages
    elif task == "clm":
        langs_scripts = get_dataset_config_names("HuggingFaceFW/fineweb-2")
        eval_languages = {}
        for ds in langs_scripts:
            lang, script = ds.split("_")
            try:
                name = ld.get(lang, tag_type=TagType.ISO_639_3_CODE).bcp_47_code
                eval_languages[(name, script)] = ds
            except KeyError:
                # print(f"Language {lang} not in database, skipping)
                continue
        print("total amount of eval langs:", len(eval_languages))
        return eval_languages


def load_eval(task, eval_language, eval_languages):
    if not task == "clm":
        dataset = load_dataset(task2ds[task], eval_languages[eval_language], trust_remote_code="True")
        if "test" in dataset.keys():
            dataset_eval = dataset["test"]
        elif "validation" in dataset.keys():
            dataset_eval = dataset["validation"]
        else:
            dataset_eval = dataset["train"]
    else:

        def generate_lines(dataset, cutoff):
            n = 0
            for example in dataset:
                if n >= cutoff:
                    break
                for line in example["text"].split("\n"):
                    if good_line := line.strip():
                        n += 1
                        # print(n, end="\r")
                        yield {"text": good_line}

        if eval_languages[eval_language] == "eng_Latn":
            ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-2", eval_languages[eval_language], streaming=True, split="train")

        dataset_eval = Dataset.from_generator(
            generate_lines, gen_kwargs={"dataset": ds, "cutoff": 2000}, split=Split.TEST
        )
    return dataset_eval


def preprocess(dataset, task, tokenizer):
    if task == "ner":

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

        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized_datasets
    elif task == "pos":

        def tokenize_and_align_labels(examples):
            tokenized = tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            all_labels = []
            for i, labels in enumerate(examples["upos"]):
                word_ids = tokenized.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                all_labels.append(label_ids)
            tokenized["labels"] = all_labels
            return tokenized

        tokenized_datasets = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized_datasets
    elif task == "copa":

        def encode_batch(examples):
            all_encoded = {"input_ids": [], "attention_mask": []}
            # Iterate through all examples in this batch
            for premise, question, choice1, choice2 in zip(
                examples["premise"], examples["question"], examples["choice1"], examples["choice2"]
            ):
                sentences_a = [premise + " " + question for _ in range(2)]
                # Both answer choices are passed in an array according to the format needed for the multiple-choice prediction head
                sentences_b = [choice1, choice2]
                encoded = tokenizer(
                    sentences_a,
                    sentences_b,
                    max_length=60,
                    truncation=True,
                    padding="max_length",
                )
                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
            return all_encoded

        def preprocess_dataset(dataset):
            # Encode the input data
            dataset = dataset.map(encode_batch, batched=True)
            # The transformers model expects the target class column to be named "labels"
            dataset = dataset.rename_column("label", "labels")
            # Transform to pytorch tensors and only output the required columns
            dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
            return dataset

        dataset_eval = preprocess_dataset(dataset)
        return dataset_eval

    elif task == "qa":

        def preprocess_validation_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=384,
                truncation="only_second",
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

            inputs["example_id"] = example_ids
            return inputs

        dataset_eval = dataset.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return dataset_eval
    elif task == "sib":
        label_list = [
            "science/technology",
            "travel",
            "politics",
            "sports",
            "health",
            "entertainment",
            "geography",
        ]
        label2id = {label: i for i, label in enumerate(label_list)}

        def preprocess_function(examples):
            # Tokenize the "text" field; returns a dict with 'input_ids', 'attention_mask', etc.
            tokens = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,  # change max_length if you need longer/shorter
            )
            # Map each category string to its integer ID
            # (when using batched=True, examples["category"] is a list of strings)
            tokens["labels"] = [label2id[cat] for cat in examples["category"]]
            return tokens

        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized_datasets

    elif task == "clm":

        def preprocess_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                # no padding here; we'll do dynamic padding in the collator
            )

        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            # remove_columns=dataset["devtest"].column_names,  # keep only model features
        )
        return tokenized_datasets


def get_compute_metrics(task, label_names=None):
    if task == "ner":
        assert label_names, "NER eval needs labels"

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            # Remove ignored index (special tokens) and convert to labels
            true_labels = [[label_names[lab] for lab in label if lab != -100] for label in labels]
            true_predictions = [
                [label_names[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metrics[task].compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
            }

    elif task == "pos":

        def compute_metrics(eval_preds):
            logits, label_ids = eval_preds
            # 1) take argmax over labels
            pred_ids = np.argmax(logits, axis=-1)

            # 2) flatten, filtering out the -100 padding label
            true_labels = []
            true_preds = []
            for pred_seq, label_seq in zip(pred_ids, label_ids):
                for pred, lab in zip(pred_seq, label_seq):
                    if lab == -100:
                        continue
                    true_labels.append(lab)
                    true_preds.append(pred)

            # 3) compute metrics
            acc = accuracy_score(true_labels, true_preds)
            # you can choose 'macro' or 'micro' or 'weighted' averages
            prec_macro = precision_score(true_labels, true_preds, average="macro", zero_division=0)
            rec_macro = recall_score(true_labels, true_preds, average="macro", zero_division=0)
            f1_macro = f1_score(true_labels, true_preds, average="macro", zero_division=0)

            prec_micro = precision_score(true_labels, true_preds, average="micro", zero_division=0)
            rec_micro = recall_score(true_labels, true_preds, average="micro", zero_division=0)
            f1_micro = f1_score(true_labels, true_preds, average="micro", zero_division=0)

            return {
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_micro": prec_micro,
                "recall_micro": rec_micro,
                "f1_micro": f1_micro,
            }
    elif task == "copa":

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": (preds == p.label_ids).mean()}

    elif task == "qa":
        n_best = 20
        max_answer_length = 30

        def compute_metrics(start_logits, end_logits, features, examples):
            example_to_features = collections.defaultdict(list)
            for idx, feature in enumerate(features):
                example_to_features[feature["example_id"]].append(idx)

            predicted_answers = []
            for example in tqdm(examples):
                example_id = example["id"]
                context = example["context"]
                answers = []

                # Loop through all features associated with that example
                for feature_index in example_to_features[example_id]:
                    start_logit = start_logits[feature_index]
                    end_logit = end_logits[feature_index]
                    offsets = features[feature_index]["offset_mapping"]

                    start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                    end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            # Skip answers that are not fully in the context
                            if offsets[start_index] is None or offsets[end_index] is None:
                                continue
                            # Skip answers with a length that is either < 0 or > max_answer_length
                            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                                continue

                            answer = {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                            answers.append(answer)

                # Select the answer with the best score
                if len(answers) > 0:
                    best_answer = max(answers, key=lambda x: x["logit_score"])
                    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
                else:
                    predicted_answers.append({"id": example_id, "prediction_text": ""})

            theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
            return metrics[task].compute(predictions=predicted_answers, references=theoretical_answers)

    elif task == "sib":

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            return metrics[task].compute(predictions=predictions, references=labels)

    elif task == "clm":

        def compute_metrics(eval_preds):
            print("if you see this, something went wrong in passing the arguments for clm")
            return {}

    return compute_metrics


def get_trainer_kwargs(task, model, tokenized_datasets, tokenizer, data_collator, compute_metrics):
    args = TrainingArguments(
        output_dir="./eval_output",
        remove_unused_columns=False if not task == "qa" else True,
        fp16=True,
        per_device_eval_batch_size=16,  # increase if you have the GPU memory
    )
    trainer_kwargs = {"model": model, "args": args, "eval_dataset": tokenized_datasets}
    if task not in ["qa", "clm"]:
        trainer_kwargs["compute_metrics"] = compute_metrics
    if task in ["qa", "sib"]:
        trainer_kwargs["processing_class"] = tokenizer
    if task != "copa":
        trainer_kwargs["data_collator"] = data_collator
    return trainer_kwargs


def get_available_adapters(local=False, model_type="xlm-roberta-base"):
    # get all adapters from huggingface
    all_models = api.list_models(author="AdapterHub", library="adapter-transformers", search=f"{model_type}-")

    to_load = {
        m.modelId: m.modelId.split(f"{model_type}-")[-1].rsplit("-wiki_pfeiffer", 1)[0]
        for m in all_models
        if m.modelId.startswith(f"AdapterHub/{model_type}-") and m.modelId.endswith("-wiki_pfeiffer")
    }
    if local:
        trained_adapter_path = "./trained_adapters/"
        for iso in local:
            to_load[trained_adapter_path + iso] = iso
        print("extended to load with locals")
    print(to_load)
    return to_load


def get_clm_adapters(local=None, convert=False):
    to_load = {}
    if local:
        # we go through all directories in the local provided path
        for adapter in os.listdir(local):
            code = adapter
            if convert:
                # convert from BCP 47 to ISO 639-3
                if "_" in code:
                    code = code.split("_")[0]
                code = ld.get(code, tag_type=TagType.ISO_639_3_CODE).bcp_47_code
            to_load[local + adapter] = code
    return to_load


def get_glots(to_load):
    manuals = {
        "Arabic": "arab1267",
        "Swahili": "swah1253",
        "Bengali": "beng1282",
        "Chinese": "mand1415",
        "Persian": "west2369",
        "Yoruba": "ilaa1246",
        "Nepali": "nepa1254",
        "Quechua": "cusc1236",
        "Estonian": "esto1258",
        "Guarani": "east2555",
    }

    glots = {}
    probs = []

    for lang in to_load.values():
        eng = ld.get(lang, tag_type=TagType.BCP_47_CODE).english_name
        glot = ld.get(lang, tag_type=TagType.BCP_47_CODE).glottocode
        # we need to find if glot is in distances
        if not glot:
            if eng in manuals.keys():
                glot = manuals[eng]
        if eng and glot:
            glots[lang] = glot
        else:
            probs.append(lang)
    if probs:
        print("no glottocodes found for these languages: ", probs)
        print("removing them from further consideration")
    for prob in probs:
        del to_load[prob]
        # happens in-place
    return glots


def merge_loaded_adapters(
    model, merge_adapter_name="joined_adapter", weights=None, delete_other=False, patterns=False, model_type="roberta"
):
    # to ensure we don't get problems, we check the config of all adapters
    all_adapters = list(model.adapters_config.adapters.keys())
    config_id = model.adapters_config.adapters[all_adapters[0]]
    config = model.adapters_config.config_map[config_id]

    for i in range(1, len(all_adapters)):
        config_id = model.adapters_config.adapters[all_adapters[i]]
        config_i = model.adapters_config.config_map[config_id]
        assert config == config_i, (
            f"Config mismatch: {config} vs {config_i}\nCurrent methodology only works for same config"
        )

    if weights is None or weights == {}:
        weights = {adapter: 1 / len(all_adapters) for adapter in all_adapters}

    if not patterns:
        if model_type in ["roberta", "bert"]:
            patterns = [
                f"{model_type}\.encoder\.layer\.(?P<one>[\d\w]+)\.output\.adapters\.(?P<adapter>\w+)\.(?P<two>\w+)(?:\.0)?\.(?P<three>\w+)",
                f"{model_type}\.invertible_adapters\.(?P<adapter>\w+)\.(?P<one>\w+)\.(?P<two>\d)\.(?P<three>\w+)",
            ]
        elif model_type == "clm":
            patterns = [
                "model\.layers\.(?P<one>[\d\w]+)\.output_adapters\.adapters\.(?P<adapter>\w+)\.(?P<two>\w+)(?:\.0)?\.(?P<three>\w+)",
                "model\.invertible_adapters\.(?P<adapter>\w+)\.(?P<one>\w+)\.(?P<two>\d)\.(?P<three>\w+)",
            ]
        else:
            raise ValueError(f"Unknown model type {model_type}")
    comp_patterns = [re.compile(pattern) for pattern in patterns]
    organized_layers = {}
    for i, pattern in enumerate(patterns):
        # we make a dictionary for each pattern
        organized_layers[i] = {}

    for key in model.state_dict().keys():
        for i, pattern in enumerate(comp_patterns):
            match = re.search(pattern, key)
            if match:
                one = match.group("one")
                two = match.group("two")
                three = match.group("three")
                adapter_name = match.group("adapter")
                if adapter_name not in weights.keys():
                    # print(f"Adapter {adapter_name} not in weights")
                    continue
                if one not in organized_layers[i]:
                    organized_layers[i][one] = {}
                if two not in organized_layers[i][one]:
                    organized_layers[i][one][two] = {}
                if three not in organized_layers[i][one][two]:
                    organized_layers[i][one][two][three] = []
                organized_layers[i][one][two][three].append((key, adapter_name))
    new_state_dict = OrderedDict()
    sd = model.state_dict()

    for i, one in organized_layers.items():
        for one, two in one.items():
            for two, three in two.items():
                for three, keys in three.items():
                    result = sum([sd[layer] * weights[adapter_name] for layer, adapter_name in keys])
                    if two == "adapter_down":
                        if model_type in ["roberta", "bert"]:
                            new_state_dict[
                                f"{model_type}.encoder.layer.{one}.output.adapters.{merge_adapter_name}.{two}.0.{three}"
                            ] = result
                        elif model_type == "clm":
                            new_state_dict[
                                f"model.layers.{one}.output_adapters.adapters.{merge_adapter_name}.{two}.0.{three}"
                            ] = result
                        else:
                            raise ValueError(f"Unknown model type {model_type}")
                    elif two == "adapter_up":
                        if model_type in ["roberta", "bert"]:
                            new_state_dict[
                                f"{model_type}.encoder.layer.{one}.output.adapters.{merge_adapter_name}.{two}.{three}"
                            ] = result
                        elif model_type == "clm":
                            new_state_dict[
                                f"model.layers.{one}.output_adapters.adapters.{merge_adapter_name}.{two}.{three}"
                            ] = result
                        else:
                            raise ValueError(f"Unknown model type {model_type}")
                    else:
                        # we are in the second pattern
                        if model_type in ["roberta", "bert"]:
                            new_state_dict[
                                f"{model_type}.invertible_adapters.{merge_adapter_name}.{one}.{two}.{three}"
                            ] = result
                        elif model_type == "clm":
                            new_state_dict[f"model.invertible_adapters.{merge_adapter_name}.{one}.{two}.{three}"] = (
                                result
                            )
                        else:
                            raise ValueError(f"Unknown model type {model_type}")

    # we now load in the new model
    if merge_adapter_name in model.adapters_config.adapters.keys():
        # remove the old one
        model.delete_adapter(merge_adapter_name)
    model.add_adapter(merge_adapter_name, config=config)
    for name, param in model.named_parameters():
        # e.g. "roberta.encoder.layer.0.output.adapters.joined_adapter.adapter_down.0.weight"
        if merge_adapter_name in name and name in new_state_dict:
            param.data.copy_(new_state_dict[name])
    if delete_other:
        for key in list(model.adapters_config.adapters.keys()):
            if key != merge_adapter_name:
                model.delete_adapter(key)

    # no need to return anything as the model is changed in place


def typological_approximation(target, glots, distance_type, limit=None):
    """
    This function takes a target language and a list of languages.
    It weights the other languages depending on their closeness to the target language.
    "distance_type" can be of following:
    "featural", "syntactic", "phonological", "geographic", "genetic", "morphological", "inventory"
    If limit is specified and is <1, it will remove all languages with a distance lower than limit.
    If limit is specified and is >=1, it works as a top-k languages filter with the highest similarity.
    """

    # 1. retrieve closeness score of all languages to target language
    weights = {}
    for lang, glot in glots.items():
        # get the distance
        try:
            # distances range from 0 to 1
            dist = 1 - u.new_distance(distance_type, [glot, target])
            # print(f"Distance {lang} to {target}: {dist}")
        except SystemExit:
            # print(f"Error: {lang} - {glot} - {target}")
            dist = 0
        except TypeError:
            print(f"No {distance_type} data for {lang} - {glot} - {target}")
            dist = 0
        weights[lang] = dist

    # Check for limit:
    if limit:
        if limit < 1:
            for lang, dist in list(weights.items()):
                if dist < limit:
                    # print(f"Removing {lang} with distance {dist}")
                    del weights[lang]
        else:  # we take the best n (limit) languages
            n = min(limit, len(weights))
            # we sort the weights
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            # print(sorted_weights)
            # we take the first n
            sorted_weights = sorted_weights[:n]
            # we convert back to dict
            weights = {k: v for k, v in sorted_weights}
    # we first check if all weights are 0:
    if sum(weights.values()) == 0:
        print("All weights are 0, returning empty dict")
        return {}
    # print(f"Weights before softmax: {weights}")
    soft_weights = torch.softmax(torch.tensor(list(weights.values())), dim=0)
    # we need to convert to list
    soft_weights = soft_weights.tolist()
    # we zippedly return the keys and normalized values
    weights = {k: v for k, v in zip(weights.keys(), soft_weights)}
    # print(f"Weights after softmax: {weights}")
    return weights


def load_finetuned_model(task):
    model_type = {
        "ner": AutoModelForTokenClassification,
        "pos": AutoModelForTokenClassification,
        "copa": AutoModelForMultipleChoice,
        "qa": AutoModelForQuestionAnswering,
        "sib": AutoModelForSequenceClassification,
    }
    model = model_type[task].from_pretrained(f"finetuned_models/xlm_{task}_finetune")
    return model
