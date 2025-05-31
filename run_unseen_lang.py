# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import (
        get_eval_languages,
        load_eval,
        preprocess,
        get_compute_metrics,
        get_trainer_kwargs,
        get_available_adapters,
        merge_loaded_adapters,
        typological_approximation,
        get_glots,
    )

    from transformers import (
        AutoTokenizer,
        HfArgumentParser,
        XLMRobertaTokenizerFast,
        DataCollatorForTokenClassification,
        DefaultDataCollator,
        DataCollatorWithPadding,
    )
    from adapters import AdapterTrainer, AutoAdapterModel
    from adapters.composition import Stack

    import os
    import json
    from qq import LanguageData, TagType

    from dataclasses import dataclass, field
    from typing import Optional, List
    import logging

    # silence urielplus INFO logs
    logging.getLogger("urielplus").setLevel(logging.ERROR)

    @dataclass
    class CustomArguments:
        """
        Arguments to direct the evaluation.

        """

        task: str = field(
            metadata={"help": ("The task to perform (ner, copa or pos)")},
        )

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

        disable_baselines: Optional[bool] = field(
            default=False,
            metadata={
                "help": (
                    "Whether to calculate the baselines. "
                    "Default True, it will calculate the baselines for the adapter combinations."
                )
            },
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
        save_adapter: Optional[bool] = field(
            default=False,
            metadata={
                "help": (
                    "Whether to save the adapter after training. "
                    "If True, it will save the adapter to the trained_adapters folder."
                )
            },
        )
        output_name: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "The name of the output file. "
                    "If not specified, it will be saved as task_eval.json in the trained_adapters folder."
                )
            },
        )

        retry: Optional[bool] = field(
            default=False,
            metadata={"help": ("Option to re-run only previously failed languages.")},
        )
        eval_override: Optional[List[str]] = field(
            default=None,
            metadata={
                "help": ("Override evaluation languages."),
                "nargs": "+",
            },
        )
        local_adapters: Optional[List[str]] = field(
            default=None, metadata={"help": ("Local language adapters to load in"), "nargs": "+"}
        )
        reverse: Optional[bool] = field(
            default=False,
            metadata={
                "help": (
                    "Whether to reverse the order of the languages. "
                    "Default False, it will keep the order of the languages."
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
            if self.retry:
                # we check if a file exists with failed languages
                if not os.path.exists(
                    f"./experiment_folder/logs/failed_languages_{self.task}{'_' + self.output_name if self.output_name else ''}.txt"
                ):
                    raise ValueError(
                        f"File with failed languages {self.task} does not exist. Please run the script without --retry first."
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

    if custom_args.distance_types_list:
        distance_types = custom_args.distance_types_list
    elif custom_args.distance_type:
        distance_types = [custom_args.distance_type]
    else:
        distance_types = ["featural"]

    task = custom_args.task
    eval_languages = get_eval_languages(task)

    if custom_args.eval_override and custom_args.eval_override[0] == "en":
        eval_languages = {"en": "en"}
    # we filter out the languages that have failed before
    # we first check if the file exists:
    if custom_args.eval_override:
        print(f"Overriding evaluation languages with {custom_args.eval_override}")
        eval_languages = {k: v for k, v in eval_languages.items() if k in custom_args.eval_override}
    elif os.path.exists(
        f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt"
    ):
        with open(
            f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
            "r",
        ) as f:
            failed_languages = f.read().splitlines()
        if custom_args.retry:
            eval_languages = {k: v for k, v in eval_languages.items() if k in failed_languages}
        else:
            eval_languages = {k: v for k, v in eval_languages.items() if k not in failed_languages}

    Tokenizer = XLMRobertaTokenizerFast if task == "pos" else AutoTokenizer
    tokenizer = Tokenizer.from_pretrained("xlm-roberta-base")
    if task == "qa":
        data_collator = DefaultDataCollator()
    elif task == "sib":
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")

    to_load = get_available_adapters(local=custom_args.local_adapters)
    for link, id in to_load.items():
        try:
            model.load_adapter(link, load_as=id)
        except OSError:
            print(f"Could not load {link}")
            continue

    if not custom_args.limit:
        limit_str = ""
        limit_p = ""
    else:
        if custom_args.limit < 1:
            decimal = str(custom_args.limit).split(".")[1]
            limit_str = f"_0{decimal}"
            limit_p = f"/0{decimal}"
        else:
            custom_args.limit = int(custom_args.limit)
            limit_str = f"_{str(custom_args.limit)}"
            limit_p = f"/{str(custom_args.limit)}"

    print(f"\n{'~' * 30}\n{task.upper()}\n{'~' * 30}")
    # We reverse the order of the languages if custom_args.reverse
    if custom_args.reverse:
        eval_languages = {k: v for k, v in reversed(eval_languages.items())}
    for eval_language in eval_languages.keys():
        print(eval_language)
        if task == "sib":
            eval_language, script = eval_languages
        else:
            script = None
        try:
            print(
                "\n\n",
                f"Evaluating {task} on {eval_language} ({ld.get(eval_language, tag_type=TagType.BCP_47_CODE).english_name})",
                f"{script}",
            )
            if custom_args.output_name:
                output_file = f"./eval_output/approximation/{eval_language}{('_' + script) if script else ''}/{task}_{custom_args.output_name}{limit_str}.json"
            else:
                output_file = f"./eval_output/approximation/{eval_language}{('_' + script) if script else ''}/{task}_eval{limit_str}.json"

            if os.path.exists(output_file):
                print(f"Skipping {eval_language} as it has already been processed. Output file: {output_file}")
                continue
            # Load and preprocess the dataset
            dataset_eval = load_eval(task, eval_language, eval_languages)
            tokenized_datasets = preprocess(dataset_eval, task, tokenizer)

            if task == "ner":
                ner_feature = dataset_eval.features["ner_tags"]
                label_names = ner_feature.feature.names
            else:
                label_names = None

            compute_metrics = get_compute_metrics(task, label_names)

            def run_eval(model, name):
                # we load in the task adapter
                if not name == task:
                    model.active_adapters = Stack(name, task)
                else:
                    model.active_adapters = name
                # prepare the common arguments
                trainer_kwargs = get_trainer_kwargs(
                    task, model, tokenized_datasets, tokenizer, data_collator, compute_metrics
                )
                # instantiate
                eval_trainer = AdapterTrainer(**trainer_kwargs)
                if task != "qa":
                    ev = eval_trainer.evaluate()
                else:
                    predictions, _, _ = eval_trainer.predict(tokenized_datasets)
                    start_logits, end_logits = predictions
                    ev = compute_metrics(start_logits, end_logits, tokenized_datasets, dataset_eval)

                print(f"Evaluation results for {name}:")
                print(ev)
                # we empty the cache and model
                # model.cpu()
                # del model
                eval_trainer = None
                # if torch.cuda.is_available():
                #    torch.cuda.empty_cache()
                # gc.collect()
                return ev

            evaluations = {}
            weights = {}
            glots = get_glots(to_load)
            if eval_language in glots.keys():
                target_glot = glots[eval_language]
            else:
                target_glot = ld.get(eval_language, tag_type=TagType.BCP_47_CODE).glottocode
            # we check if the adapter has already been created before
            for distance_type in distance_types:
                adapter_name = f"reconstructed_{eval_language}_{distance_type}{limit_str}"
                adapter_path = f"./trained_adapters/typological/{eval_language}/{distance_type}_extended_{limit_p}"
                if custom_args.output_name:
                    adapter_path = (
                        f"./trained_adapters/typological/{eval_language}/{distance_type}_{custom_args.output_name}"
                    )
                weights[distance_type] = {}
                if os.path.exists(adapter_path):
                    print("Adapter already exists, loading instead")
                    model.load_adapter(
                        adapter_path,
                        load_as=adapter_name,
                    )

                else:
                    weights[distance_type] = typological_approximation(
                        target_glot, glots, distance_type, custom_args.limit
                    )
                    if weights == {}:
                        print(f"No adapters found for {eval_language} with distance type {distance_type}")
                        continue
                    merge_loaded_adapters(
                        model, merge_adapter_name=adapter_name, weights=weights[distance_type], delete_other=False
                    )
                    # save this adapter
                    # check if directory exists first
                    if custom_args.save_adapter:
                        if not os.path.exists(adapter_path):
                            os.makedirs(adapter_path)
                        model.save_adapter(adapter_path, adapter_name)
                model.load_adapter(f"./trained_adapters/task_adapters/{task}", load_as=task)
                print(f"evaluating on reconstructed {eval_language} adapter, distance type {distance_type}")
                evaluations["reconstructed_" + distance_type] = run_eval(model, adapter_name)
                model.delete_adapter(adapter_name)
                # delete the adapter for further iterations
                model.delete_adapter(task)

            if not custom_args.disable_baselines:
                model.load_adapter(f"./trained_adapters/task_adapters/{task}", load_as=task)
                # we calculate the baseline of using the english language model and the task adapter
                print("evaluating on baseline (english model + task adapter)")
                evaluations["baseline_en"] = run_eval(model, "en")
                model.delete_adapter(task)
                model.load_adapter(f"./trained_adapters/task_adapters/{task}", load_as=task)
                print("evaluating on baseline (only task adapter")
                # we calculate a baseline (just task adapter)
                evaluations["baseline_task_adapter"] = run_eval(model, task)
                # we calculate a baseline (just average over all adapter)
                # we load the mono/huge_avg_adapter for this
                print("evaluating on baseline (non-weighted average)")
                model.load_adapter("./trained_adapters/typological/huge_avg_adapter", load_as="huge_avg_adapter")
                evaluations["baseline_avg_adapter"] = run_eval(model, "huge_avg_adapter")
                model.delete_adapter("huge_avg_adapter")
                # we calculate the baseline of using the typologically closest model and the task adapter
                print("evaluating on baseline (closest model + task adapter)")
                for distance_type in distance_types:
                    try:
                        # we have to calculate these if we skipped the adapter creation
                        # we set limit to one so we only get the best adapter
                        if weights[distance_type] == {}:
                            weights[distance_type] = typological_approximation(target_glot, glots, distance_type, 1)
                        # we load the closest adapter
                        closest_adapter = max(weights[distance_type], key=weights[distance_type].get)
                        print(
                            f"closest {distance_type} adapter is {closest_adapter} ({ld.get(closest_adapter, tag_type=TagType.BCP_47_CODE).english_name})"
                        )
                        evaluations[f"baseline_closest_{distance_type}"] = run_eval(model, closest_adapter)
                    except Exception as e:
                        print(f"Error finding closest adapter: {e}")
                model.delete_adapter(task)

                # we calculate the No Train but Gain baseline (english + closest adapter)
                # for this, we retrieve the closest available adapter that is NOT english OR the language itself
                # we do this to ensure fair comparison: the basleine of using the language itself is already included
                try:
                    if "featural" in weights.keys() and len(weights["featural"]) >= 3:
                        train_gain = weights["featural"]
                    else:
                        train_gain = typological_approximation(target_glot, glots, "featural", 3)
                        # we select the highest that is not the language itself (eval_language) or english (en)
                    if eval_language in train_gain.keys():
                        train_gain[eval_language] = 0
                    if "en" in train_gain.keys():
                        train_gain["en"] = 0
                    related = max(train_gain, key=train_gain.get)
                    print(f"calculating no train but gain baseline with closest adapter {related}")
                    # as no preferred value for lambda is found by Klimaszewski, we do equal weighting for en and related
                    merge_loaded_adapters(
                        model, merge_adapter_name="no_train_gain", weights={"en": 0.5, related: 0.5}, delete_other=False
                    )
                    model.load_adapter(f"./trained_adapters/task_adapters/{task}", load_as=task)
                    evaluations["no_train_gain"] = run_eval(model, "no_train_gain")
                    # we now delete the added adapters
                    model.delete_adapter("no_train_gain")
                    model.delete_adapter(task)
                except Exception as e:
                    print(f"Error calculating no train but gain baseline: {e}")
                    # we print the active adapters: the model layers
                    print(model.roberta.encoder.layer[0].output.adapters)
                    continue

            if not os.path.exists(f"./eval_output/approximation/{eval_language}"):
                os.makedirs(f"./eval_output/approximation/{eval_language}")
            # we save this
            if custom_args.output_name:
                output_file = (
                    f"./eval_output/approximation/{eval_language}/{task}_{custom_args.output_name}{limit_str}.json"
                )
            else:
                output_file = f"./eval_output/approximation/{eval_language}/{task}_eval{limit_str}.json"
            with open(output_file, "w") as f:
                json.dump(evaluations, f, indent=4)
                print("Saved evaluations to file")
            # we write the language name to "done languages"
            # with open(done_file, "a") as f:
            #    f.write(f"{eval_language}\n")
        except RuntimeError as e:
            print(f"RuntimeError {e}, skipping this language")
            # we write this language to a file so we do not check it again
            with open(
                f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
                "a",
            ) as f:
                f.write(f"{eval_language}\n")
            continue
        except IndexError as e:
            print(f"IndexError {e}, skipping this language")
            with open(
                f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
                "a",
            ) as f:
                f.write(f"{eval_language}\n")
            continue
        except KeyError as e:
            with open(
                f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
                "a",
            ) as f:
                f.write(f"{eval_language}\n")
            print(f"KeyError {e}, skipping this language")


if __name__ == "__main__":
    debug = True
    job_name = debug * "debug_" + "unseen_task"

    master_dir = find_master()

    # Set the experi    ment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = f"gpu_p100{debug * '_debug'}"
    # some shenanigans to pass a time argument through submitit
    first = sys.argv[1]
    if first.startswith("--"):
        job_input = sys.argv[1:]
        time = "02:30:00"
    else:
        job_input = sys.argv[2:]
        if len(first) == 1:
            time = f"0{first}:00:00"
        else:
            time = first
    parameters = {
        "slurm_partition": partition,
        # "slurm_time": "03:00:00",
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else time}",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": f"{'genius' if partition.startswith(('gpu_p100', 'gpu_v100')) else 'wice'}",
            "account": os.environ["ACCOUNT_INFO"],  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": f"{'' if partition.endswith('debug') else 'stef.accou@student.kuleuven.be'}",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    # job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    print("job submitted")
