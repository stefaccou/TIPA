# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import get_available_adapters, merge_loaded_adapters, typological_approximation, get_glots

    from transformers import TrainingArguments, HfArgumentParser, XLMRobertaTokenizerFast
    from adapters import AdapterTrainer, AutoAdapterModel
    from adapters.composition import Stack

    from datasets import load_dataset, get_dataset_config_names
    from transformers import DataCollatorForTokenClassification
    import os
    import gc
    import json
    import torch
    import numpy as np
    import random
    from urielplus import urielplus
    from qq import LanguageData, TagType
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
        limit: Optional[float] = field(
            default=None,
            metadata={
                "help": (
                    "The limit for the distance types. If <1, it will remove all languages with a distance score lower than limit. "
                    "If >=1, it works as a top-k languages filter with the highest similarity."
                )
            },
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

    ud_datasets = get_dataset_config_names("universal_dependencies")
    eval_languages_dict = {}
    for ds in ud_datasets:
        lang = ds.split("_")[0]
        try:
            ld.get(lang, tag_type=TagType.BCP_47_CODE)
            eval_languages_dict[lang] = ds
        except KeyError:
            # print(f"Language {lang} not in database, skipping")
            continue
    eval_languages = list(eval_languages_dict.keys())

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
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
            dataset_eval = load_dataset(
                "universal_dependencies", eval_languages_dict[eval_language], trust_remote_code=True
            )
            print(dataset_eval.keys())

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

            if "test" in dataset_eval.keys():
                dataset_eval = dataset_eval["test"]
            elif "validation" in dataset_eval.keys():
                dataset_eval = dataset_eval["validation"]
            else:
                dataset_eval = dataset_eval["train"]

            # apply to train/validation
            tokenized_datasets = dataset_eval.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset_eval.column_names,
            )

            def compute_metrics(eval_preds):
                """
                eval_preds: tuple(logits, label_ids)
                  - logits: np.array of shape (batch_size, seq_len, num_labels)
                  - label_ids: np.array of shape (batch_size, seq_len)
                """
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

            def run_eval(model, name):
                # we load in the task adapter
                if not name == "ud_pos":
                    model.active_adapters = Stack(name, "ud_pos")
                else:
                    model.active_adapters = name
                # we check if the dataset has a test set or a validation set

                eval_trainer = AdapterTrainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir="./eval_output",
                        remove_unused_columns=False,
                    ),
                    data_collator=data_collator,
                    eval_dataset=tokenized_datasets,
                    compute_metrics=compute_metrics,
                )
                ev = eval_trainer.evaluate()
                print(f"Evaluation results for {name}:")
                print(ev)
                # we empty the cache
                model.cpu()
                del model
                del eval_trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                return ev

            evaluations = {}
            weights_dist = {}
            # we check if the adapter has already been created before
            for distance_type in distance_types:
                adapter_name = f"reconstructed_{eval_language}_{distance_type}{'_' + str(custom_args.limit) if custom_args.limit else ''}"
                adapter_path = f"./trained_adapters/typological/{eval_language}/{distance_type}{'/' + str(custom_args.limit) if custom_args.limit else ''}"
                weights_dist[distance_type] = {}
                if os.path.exists(adapter_path):
                    print("Adapter already exists, loading instead")
                    model.load_adapter(
                        adapter_path,
                        load_as=adapter_name,
                    )

                else:
                    target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
                    print(f"target glot found: {target_glot}")
                    weights_dist[distance_type] = typological_approximation(
                        target_glot, get_glots(to_load), distance_type, custom_args.limit
                    )
                    # print("active adapters to be merged:")
                    # print(model.roberta.encoder.layer[0].output.adapters)
                    merge_loaded_adapters(
                        model, merge_adapter_name=adapter_name, weights=weights_dist[distance_type], delete_other=False
                    )
                    # save this adapter
                    # check if directory exists first
                    if not os.path.exists(adapter_path):
                        os.makedirs(adapter_path)
                    model.save_adapter(adapter_path, adapter_name)
                model.load_adapter("./trained_adapters/ud_pos", load_as="ud_pos")
                print(f"evaluating on reconstructed {eval_language} adapter, distance type {distance_type}")
                evaluations["reconstructed_" + distance_type] = run_eval(model, adapter_name)
                model.delete_adapter(adapter_name)
                # delete the adapter for further iterations
                model.delete_adapter("ud_pos")

            model.load_adapter("./trained_adapters/ud_pos", load_as="ud_pos")
            print("evaluating on baseline (only task adapter")
            # we calculate a baseline (just ud_pos adapter)
            evaluations["baseline_ud_pos"] = run_eval(model, "ud_pos")

            # we calculate a baseline (just average over all adapter)
            # we load the mono/huge_avg_adapter for this
            print("evaluating on baseline (non-weighted average)")
            model.load_adapter("./trained_adapters/typological/huge_avg_adapter", load_as="huge_avg_adapter")
            evaluations["baseline_avg_adapter"] = run_eval(model, "huge_avg_adapter")

            # we calculate the baseline of using the english language model and the ud_pos adapter
            print("evaluating on baseline (english model + ud_pos adapter)")
            evaluations["baseline_en_ud_pos"] = run_eval(model, "en")  # en is in the list of available adapters

            # we calculate the baseline of using the typologically closest model and the ud_pos adapter
            print("evaluating on baseline (closest model + ud_pos adapter)")
            for distance_type in distance_types:
                try:
                    # we have to calculate these if we skipped the adapter creation
                    # we set limit to one so we only get the best adapter
                    if distance_type not in weights_dist.keys():
                        target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
                        weights_dist[distance_type] = typological_approximation(
                            target_glot, get_glots(to_load), distance_type, 1
                        )

                    # we load the closest adapter
                    closest_adapter = max(weights_dist[distance_type], key=weights_dist[distance_type].get)
                    print(
                        f"closest {distance_type} adapter is {closest_adapter} ({ld.get(closest_adapter, tag_type=TagType.BCP_47_CODE).english_name})"
                    )
                    evaluations["baseline_closest_ud_pos"] = run_eval(model, closest_adapter)
                except Exception as e:
                    print(f"Error finding closest adapter: {e}")

            # we delete the added adapters
            model.delete_adapter("huge_avg_adapter")
            model.delete_adapter("ud_pos")

            # we save this
            with open(
                f"./trained_adapters/typological/{eval_language}/pos_eval{'_' + str(custom_args.limit) if custom_args.limit else ''}",
                "w",
            ) as f:
                json.dump(evaluations, f, indent=4)
                print("Saved evaluations to file")
            # we write the language name to "done languages"
            # with open(done_file, "a") as f:
            #    f.write(f"{eval_language}\n")
        except RuntimeError as e:
            print(f"RuntimeError {e}, skipping this language")
            # we write this language to a file so we do not check it again
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            continue
        except IndexError as e:
            print(f"IndexError {e}, skipping this language")
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            continue
        except KeyError as e:
            # with open(failed_file_template.format(distance_type), "a") as f:
            #    f.write(f"{eval_language}\n")
            print(f"KeyError {e}, (qq unseen language) skipping this language")


if __name__ == "__main__":
    job_name = "debug_unseen_ud_pos"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_p100_debug",
        "slurm_time": "01:00:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "genius",
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
