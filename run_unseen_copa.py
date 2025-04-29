# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import get_available_adapters, merge_loaded_adapters, typological_approximation, get_glots

    from transformers import TrainingArguments, AutoTokenizer, EvalPrediction, HfArgumentParser
    from adapters import AdapterTrainer, AutoAdapterModel
    from adapters.composition import Stack

    from datasets import load_dataset, get_dataset_config_names
    import os
    import gc
    import json
    import torch
    import numpy as np
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

    # we load in the dataset names
    eval_languages = get_dataset_config_names("xcopa")
    eval_languages = [lan for lan in eval_languages if len(lan) <= 3]
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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
            dataset_eval = load_dataset("xcopa", eval_language, trust_remote_code=True)

            def encode_batch(examples):
                """Encodes a batch of input data using the model tokenizer."""
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

            def compute_accuracy(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                return {"acc": (preds == p.label_ids).mean()}

            def run_eval(model, name):
                # we load in the task adapter
                if not name == "copa":
                    model.active_adapters = Stack(name, "copa")
                else:
                    model.active_adapters = name
                eval_trainer = AdapterTrainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir="./eval_output",
                        remove_unused_columns=False,
                    ),
                    eval_dataset=dataset_eval["test"],
                    compute_metrics=compute_accuracy,
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

            # we preprocess the dataset
            dataset_eval = preprocess_dataset(dataset_eval)

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
                model.load_adapter("./trained_adapters/copa", load_as="copa")
                print(f"evaluating on reconstructed {eval_language} adapter, distance type {distance_type}")
                evaluations["reconstructed_" + distance_type] = run_eval(model, adapter_name)
                model.delete_adapter(adapter_name)
                # delete the adapter for further iterations
                model.delete_adapter("copa")

            # load in copa adapter AFTER creation of merged adapter
            model.load_adapter("./trained_adapters/copa", load_as="copa")
            print("evaluating on baseline (only task adapter")
            # we calculate a baseline (just copa adapter)
            evaluations["baseline_copa"] = run_eval(model, "copa")

            # we calculate a baseline (just average over all adapter)
            # we load the mono/huge_avg_adapter for this
            print("evaluating on baseline (non-weighted average)")
            model.load_adapter("./trained_adapters/typological/huge_avg_adapter", load_as="huge_avg_adapter")
            evaluations["baseline_avg_adapter"] = run_eval(model, "huge_avg_adapter")

            # we calculate the baseline of using the english language model and the ner adapter
            print("evaluating on baseline (english model + copa adapter)")
            evaluations["baseline_en_copa"] = run_eval(model, "en")  # en is in the list of available adapters

            # we calculate the baseline of using the typologically closest model and the ner adapter
            print("evaluating on baseline (closest model + copa adapter)")

            # we calculate the baseline of using the typologically closest model and the ner adapter
            print("evaluating on baseline (closest model + ner adapter)")
            for distance_type in distance_types:
                try:
                    # we have the adapters, and weights
                    adapters_weights = {}
                    # we have to calculate these if we skipped the adapter creation
                    if not weights[distance_type]:
                        target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
                        weights[distance_type] = typological_approximation(
                            target_glot, get_glots(to_load), distance_type
                        )

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
            model.delete_adapter("copa")

            # we save this
            with open(f"./trained_adapters/typological/{eval_language}/copa_eval.json", "w") as f:
                json.dump(evaluations, f, indent=4)
                print("Saved evaluations to file")

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
    job_name = "unseen_copa"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_h100",
        "slurm_time": "01:00:00",
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
    # job_input = sys.argv[1] if len(sys.argv) > 1 else "default text"
    job = executor.submit(main, job_input)
    # job = executor.submit(main)
    print("job submitted")
