# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import get_available_adapters, merge_loaded_adapters, typological_approximation, get_glots

    from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
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
    import random
    from urielplus import urielplus
    from qq import LanguageData, TagType

    from dataclasses import dataclass, field
    from typing import Optional, List

    @dataclass
    class CustomArguments:
        """
        Arguments to direct the evaluation.

        """

        distance_types_list: Optional[List[str]] = field(
            default=None,
            metadata={
                "help": ("The distance types to be used for typological approximation. "),
                "nargs": "+",
                "choices": [
                    "featural",
                    "syntactic",
                    "phonological",
                    "geographic",
                    "genetic",
                    "morphological",
                    "inventory",
                ],
            },
        )
        distance_type: Optional[str] = field(
            default=None,
            metadata={"help": ("The distance type to be used for typological approximation. ")},
        )
        iterations: Optional[int] = field(
            default=None,
            metadata={"help": ("The number of iterations to be run. ")},
        )

        def __post_init__(self):
            # we check if distance is in the list of available distances
            # OR a list combination of these types
            if self.distance_type and self.distance_type not in [
                "featural",
                "syntactic",
                "phonological",
                "geographic",
                "genetic",
                "morphological",
                "inventory",
            ]:
                raise ValueError(
                    f"Distance type {self.distance_type} not in featural, syntactic, phonological, geographic, genetic, morphological, inventory"
                )

    parser = HfArgumentParser(CustomArguments)
    # we remove sys.argv as it interferes with parsing
    sys.argv = ""
    if len(submit_arguments) == 1 and submit_arguments[0].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        custom_args = parser.parse_json_file(json_file=os.path.abspath(submit_arguments[0]))
    else:
        print("calling parser")
        # add a comma to refer to first part of tuple output
        (custom_args,) = parser.parse_args_into_dataclasses(submit_arguments)

    print("custom args: ", custom_args)

    metric = evaluate.load("seqeval")
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

    if custom_args.distance_types_list:
        distance_types = custom_args.distance_types_list
    elif custom_args.distance_type:
        distance_types = [custom_args.distance_type]
    else:
        distance_types = ["featural"]

    eval_languages = get_dataset_config_names("unimelb-nlp/wikiann")
    eval_languages = [lan for lan in eval_languages if len(lan) <= 3]

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
        iterations = int(custom_args.iterations)
        print(f"Number of iterations given: {iterations}")
    except (ValueError, TypeError):
        iterations = len(eval_languages)
        print(f"No iterations given, going for all remaining ({iterations}) languages in dataset")
    if iterations == 0:
        print("No iterations given, exiting")
        return
    for i in range(iterations):
        eval_language = random.choice(eval_languages)
        eval_languages.remove(eval_language)

        try:
            print(
                "\n\n",
                f"Evaluating on randomly chosen language {eval_language} ({ld.get(eval_language, tag_type=TagType.BCP_47_CODE).english_name})",
            )
            dataset_eval = load_dataset("wikiann", eval_language, trust_remote_code=True)

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

            def compute_metrics(eval_preds):
                logits, labels = eval_preds
                predictions = np.argmax(logits, axis=-1)

                # Remove ignored index (special tokens) and convert to labels
                true_labels = [[label_names[lab] for lab in label if lab != -100] for label in labels]
                true_predictions = [
                    [label_names[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
                return {
                    "precision": all_metrics["overall_precision"],
                    "recall": all_metrics["overall_recall"],
                    "f1": all_metrics["overall_f1"],
                    "accuracy": all_metrics["overall_accuracy"],
                }

            def run_eval(model, name):
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

            evaluations = {}
            weights = {}
            # we check if the adapter has already been created before
            for distance_type in distance_types:
                adapter_name = f"reconstructed_{eval_language}_{distance_type}"
                adapter_path = f"./trained_adapters/typological/{eval_language}/{distance_type}"
                weights[distance_type] = {}
                if os.path.exists(adapter_path):
                    print("Adapter already exists, loading instead")
                    model.load_adapter(
                        adapter_path,
                        load_as=adapter_name,
                    )

                else:
                    target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
                    weights[distance_type] = typological_approximation(target_glot, get_glots(to_load), distance_type)

                    merge_loaded_adapters(
                        model, merge_adapter_name=adapter_name, weights=weights[distance_type], delete_other=False
                    )
                    # save this adapter
                    # check if directory exists first
                    if not os.path.exists(adapter_path):
                        os.makedirs(adapter_path)
                    model.save_adapter(adapter_path, adapter_name)
                model.load_adapter("./trained_adapters/ner", load_as="ner")
                print(f"evaluating on reconstructed {eval_language} adapter, distance type {distance_type}")
                evaluations["reconstructed_" + distance_type] = run_eval(model, adapter_name)
                model.delete_adapter(adapter_name)
                # delete the adapter for further iterations
                model.delete_adapter("ner")

            model.load_adapter("./trained_adapters/ner", load_as="ner")
            print("evaluating on baseline (only task adapter")
            # we calculate a baseline (just ner adapter)
            evaluations["baseline_ner"] = run_eval(model, "ner")

            # we calculate a baseline (just average over all adapter)
            # we load the mono/huge_avg_adapter for this
            print("evaluating on baseline (non-weighted average)")
            model.load_adapter("./trained_adapters/typological/huge_avg_adapter", load_as="huge_avg_adapter")
            evaluations["baseline_avg_adapter"] = run_eval(model, "huge_avg_adapter")

            # we calculate the baseline of using the english language model and the ner adapter
            print("evaluating on baseline (english model + ner adapter)")
            evaluations["baseline_en_ner"] = run_eval(model, "en")  # en is in the list of available adapters

            # we calculate the baseline of using the typologically closest model and the ner adapter
            print("evaluating on baseline (closest model + ner adapter)")
            for distance_type in distance_types:
                try:
                    # we have the adapters, and weights
                    adapters_weights = {}
                    # we have to calculate these if we skipped the adapter creation
                    if not weights[distance_type]:
                        target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
                        weights[distance_type] = typological_approximation(target_glot, get_glots(to_load))

                    for adapter, weight in zip(to_load.values(), weights[distance_type]):
                        adapters_weights[adapter] = weight
                    # we load the closest adapter
                    closest_adapter = max(adapters_weights, key=adapters_weights.get)
                    print(
                        f"closest {distance_type} adapter is {closest_adapter} ({ld.get(closest_adapter, tag_type=TagType.BCP_47_CODE).english_name})"
                    )
                    evaluations["baseline_closest_ner"] = run_eval(model, closest_adapter)
                except Exception as e:
                    print(f"Error finding closest adapter: {e}")

            # we delete the added adapters
            model.delete_adapter("huge_avg_adapter")
            model.delete_adapter("ner")

            # we save this
            with open(f"./trained_adapters/typological/{eval_language}/ner_eval.json", "w") as f:
                json.dump(evaluations, f, indent=4)
                print("Saved evaluations to file")
            # we write the language name to "done languages"
            # with open(done_file, "a") as f:
            #    f.write(f"{eval_language}\n")
        except RuntimeError:
            print("RuntimeError, skipping this language")
            # we write this language to a file so we do not check it again
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            continue
        except IndexError:
            print("IndexError, skipping this language")
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            continue
        except KeyError:
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            print("KeyError, (qq unseen language) skipping this language")


if __name__ == "__main__":
    job_name = "unseen_ner"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_h100",
        "slurm_time": "03:30:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": os.environ["ACCOUNT_INFO"],  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "stef.accou@student.kuleuven.be",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    # job = executor.submit(main)
    print("job submitted")
