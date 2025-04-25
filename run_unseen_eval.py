# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from custom_submission_utils import find_master, update_submission_log


def main(job_input):
    from unseen_eval import get_available_adapters, merge_loaded_adapters, typological_approximation, get_glots

    from transformers import TrainingArguments, AutoTokenizer
    from adapters import AdapterTrainer, AutoAdapterModel
    from adapters.composition import Stack

    from datasets import load_dataset, get_dataset_config_names
    from transformers import DataCollatorForTokenClassification
    import os
    import gc
    import json
    import torch
    import numpy as np
    import evaluate

    metric = evaluate.load("seqeval")
    import random

    from huggingface_hub import HfApi

    api = HfApi()

    from qq import LanguageData, TagType

    ld = LanguageData.from_db()

    from urielplus import urielplus

    u = urielplus.URIELPlus()
    u.set_cache(True)
    try:
        u.integrate_grambank()
    except:
        print("already using GramBank")
    try:
        u.set_glottocodes()
    except:
        print("already using Glottocodes")

    eval_languages = get_dataset_config_names("unimelb-nlp/wikiann")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")

    to_load = get_available_adapters()
    for link, id in to_load.items():
        try:
            model.load_adapter(link, load_as=id)
        except OSError:
            print(f"Could not load {link}")
            continue

    try:
        iterations = int(job_input)
    except:
        print("No iterations given, using default of 1")
        iterations = 1
    for i in range(iterations):
        eval_language = random.choice([lan for lan in eval_languages if len(lan) <= 3])
        print(
            f"Evaluating on randomly chosen language {eval_language} ({ld.get(eval_language, tag_type=TagType.BCP_47_CODE).english_name})"
        )

        dataset_eval = load_dataset("wikiann", eval_language, trust_remote_code=True)
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

        # Load and preprocess WikiANN

        tokenized_datasets = dataset_eval.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_eval["train"].column_names,
        )

        # dataset_en is now ready to be used with adapters for cross-lingual transfer

        ner_feature = dataset_eval["train"].features["ner_tags"]

        label_names = ner_feature.feature.names
        words = dataset_eval["train"][0]["tokens"]
        labels = dataset_eval["train"][0]["ner_tags"]
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
            }

        def eval(model, name):
            # we load in the task adapter
            if not name == "ner":
                model.active_adapters = Stack(name, "ner")
            else:
                model.active_adapters = name

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
            print(f"Evaluation results for {name}:")
            print(ev)
            # we empty the cache and model
            model.cpu()
            del model
            del eval_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return ev

        # we check if the adapter has already been created before
        if os.path.exists(f"./trained_adapters/typological/{eval_language}"):
            print("Adapter already exists, loading instead")
            model.load_adapter(
                f"./trained_adapters/typological/{eval_language}", load_as="reconstructed_" + eval_language
            )
        else:
            target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
            weights = typological_approximation(target_glot, get_glots(to_load))

            merge_loaded_adapters(
                model, merge_adapter_name=f"reconstructed_{eval_language}", weights=weights, delete_other=False
            )
            # save this adapter
            # check if directory exists first
            if not os.path.exists(f"./trained_adapters/typological/{eval_language}"):
                os.makedirs(f"./trained_adapters/typological/{eval_language}")
            model.save_adapter(f"./trained_adapters/typological/{eval_language}", "reconstructed_" + eval_language)

        model.load_adapter("./trained_adapters/ner", load_as="ner")

        evaluations = {}
        evaluations["reconstructed_" + eval_language] = eval(model, f"reconstructed_{eval_language}")

        # we calculate a baseline (just ner adapter)
        evaluations["baseline_ner"] = eval(model, "ner")

        # we calculate a baseline (just average over all adapter)
        # we load the mono/huge_avg_adapter for this
        model.load_adapter("./trained_adapters/typological/huge_avg_adapter", load_as="huge_avg_adapter")
        evaluations["baseline_avg_adapter"] = eval(model, "huge_avg_adapter")
        # we delete the added adapters
        model.delete_adapter("huge_avg_adapter")
        model.delete_adapter("ner")
        model.delete_adapter("reconstructed_" + eval_language)

        # we save this
        with open(f"./trained_adapters/typological/{eval_language}/eval.json", "w") as f:
            json.dump(evaluations, f, indent=4)
            print("Saved evaluations to file")


if __name__ == "__main__":
    job_name = "unseen_eval_random"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_a100",
        "slurm_time": "01:05:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": os.environ["ACCOUNT_INFO"],  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    # job = executor.submit(main)
    print("job submitted")
