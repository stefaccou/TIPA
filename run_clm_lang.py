# This script will load in all existing adapters from the huggingface library and re-construct an unseen language adapter
import submitit
import os
import sys
from cluster_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import (
        get_eval_languages,
        load_eval,
        preprocess,
        get_compute_metrics,
        get_trainer_kwargs,
        get_clm_adapters,
        merge_loaded_adapters,
        typological_approximation,
        get_glots,
    )
    from datasets.exceptions import DatasetGenerationError

    from transformers import (
        Trainer,
        AutoModelForCausalLM,
        AutoTokenizer,
        HfArgumentParser,
        DataCollatorForLanguageModeling,
    )
    from adapters import AdapterTrainer, init

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
        local_adapters: Optional[str] = field(default=None, metadata={"help": ("Local language adapters to load in")})
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

    task = "clm"
    eval_languages = get_eval_languages(task)

    # we filter out the languages that have failed before
    # we first check if the file exists:
    if custom_args.eval_override:
        print(f"Overriding evaluation languages with {custom_args.eval_override}")
        eval_languages = {k: v for k, v in eval_languages.items() if k in custom_args.eval_override}

    # We remove this to do all languages always
    # elif os.path.exists(
    #     f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt"
    # ):
    #     with open(
    #         f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
    #         "r",
    #     ) as f:
    #         failed_languages = f.read().splitlines()
    #     if custom_args.retry:
    #         eval_languages = {k: v for k, v in eval_languages.items() if k in failed_languages}
    #     else:
    #         eval_languages = {k: v for k, v in eval_languages.items() if k not in failed_languages}

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    # We use a CLM data collator: pads dynamically and sets labels=input_ids with -100 on padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt")
    init(model)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Gemma does this
        model.config.pad_token_id = tokenizer.pad_token_id

    to_load = get_clm_adapters(local=custom_args.local_adapters, convert=True)
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
        # print(eval_language)
        if task in ["sib", "clm"]:
            eval_lang, script = eval_language
        else:
            eval_lang = eval_language
            script = None
        try:
            print(
                "\n\n",
                f"Evaluating {task} on {eval_lang} ({ld.get(eval_lang, tag_type=TagType.BCP_47_CODE).english_name})",
                f"{script}",
            )
            if custom_args.output_name:
                output_file = f"./eval_output/clm/{eval_lang}/{task}{f'_{script}' if script else ''}_{custom_args.output_name}{limit_str}.json"
            else:
                output_file = (
                    f"./eval_output/clm/{eval_lang}/{task}{f'_{script}' if script else ''}_eval{limit_str}.json"
                )

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
                model.active_adapters = name
                # prepare the common arguments
                trainer_kwargs = get_trainer_kwargs(
                    task, model, tokenized_datasets, tokenizer, data_collator, compute_metrics
                )
                # instantiate
                if name is not None:
                    eval_trainer = AdapterTrainer(**trainer_kwargs)
                else:
                    eval_trainer = Trainer(**trainer_kwargs)
                if task != "qa":
                    ev = eval_trainer.evaluate()
                else:
                    predictions, _, _ = eval_trainer.predict(tokenized_datasets)
                    start_logits, end_logits = predictions
                    ev = compute_metrics(start_logits, end_logits, tokenized_datasets, dataset_eval)

                # print(f"Evaluation results for {name}:")
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
            if eval_lang in glots.keys():
                target_glot = glots[eval_lang]
            else:
                target_glot = ld.get(eval_lang, tag_type=TagType.BCP_47_CODE).glottocode
            # we check if the adapter has already been created before
            for distance_type in distance_types:
                adapter_name = f"reconstructed_{eval_lang}_{distance_type}{limit_str}"
                adapter_path = f"./trained_adapters/clm/{eval_lang}/{distance_type}_extended_{limit_p}"
                if custom_args.output_name:
                    adapter_path = f"./trained_adapters/clm/{eval_lang}/{distance_type}_{custom_args.output_name}"
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
                    print(weights)
                    if weights[distance_type] == {}:
                        print(f"No adapters found for {eval_lang} with distance type {distance_type}")
                        raise ValueError("No adapters found")

                    merge_loaded_adapters(
                        model, merge_adapter_name=adapter_name, weights=weights[distance_type], delete_other=False
                    )
                    # save this adapter
                    # check if directory exists first
                    if custom_args.save_adapter:
                        if not os.path.exists(adapter_path):
                            os.makedirs(adapter_path)
                        model.save_adapter(adapter_path, adapter_name)

                print(f"Evaluating on reconstructed {eval_lang} adapter, distance type {distance_type}")
                evaluations["reconstructed_" + distance_type] = run_eval(model, adapter_name)
                model.delete_adapter(adapter_name)
                # delete the adapter for further iterations

            if not custom_args.disable_baselines:
                # we calculate the baseline of using the english language model and the task adapter
                print("Evaluating base model without adapter")
                evaluations["baseline_en"] = run_eval(model, None)

                # we calculate a baseline (just average over all adapter)
                # we load the mono/huge_avg_adapter for this
                # print("evaluating on baseline (non-weighted average)")
                # model.load_adapter("./trained_adapters/clm/huge_avg_adapter", load_as="huge_avg_adapter")
                # evaluations["baseline_avg_adapter"] = run_eval(model, "huge_avg_adapter")
                # model.delete_adapter("huge_avg_adapter")
                # we calculate the baseline of using the typologically closest model and the task adapter
                print("evaluating on baseline (closest adapter)")
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

                # we calculate the No Train but Gain baseline (english + closest adapter)
                # for this, we retrieve the closest available adapter that is NOT english OR the language itself
                # we do this to ensure fair comparison: the basleine of using the language itself is already included
                # try:
                #     if "featural" in weights.keys() and len(weights["featural"]) >= 3:
                #         train_gain = weights["featural"]
                #     else:
                #         train_gain = typological_approximation(target_glot, glots, "featural", 3)
                #         # we select the highest that is not the language itself (eval_language) or english (en)
                #     if eval_lang in train_gain.keys():
                #         train_gain[eval_lang] = 0
                #     if "en" in train_gain.keys():
                #         train_gain["en"] = 0
                #     related = max(train_gain, key=train_gain.get)
                #     print(f"calculating no train but gain baseline with closest adapter {related}")
                #     # as no preferred value for lambda is found by Klimaszewski, we do equal weighting for en and related
                #     merge_loaded_adapters(
                #         model, merge_adapter_name="no_train_gain", weights={"en": 0.5, related: 0.5}, delete_other=False
                #     )
                #     evaluations["no_train_gain"] = run_eval(model, "no_train_gain")
                #     # we now delete the added adapters
                #     model.delete_adapter("no_train_gain")
                # except Exception as e:
                #     print(f"Error calculating no train but gain baseline: {e}")
                #     # we print the active adapters: the model layers
                #     # print(model.roberta.encoder.layer[0].output.adapters)
                #     continue

            if not os.path.exists(f"./eval_output/clm/{eval_lang}"):
                os.makedirs(f"./eval_output/clm/{eval_lang}")
            # we save this
            with open(output_file, "w") as f:
                json.dump(evaluations, f, indent=4)
                print("Saved evaluations to file")
            # we write the language name to "done languages"
            # with open(done_file, "a") as f:
            #    f.write(f"{eval_lang}\n")
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
        except DatasetGenerationError as e:
            print(f"DatasetGenerationError {e}, skipping this language")
            with open(
                f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
                "a",
            ) as f:
                f.write(f"{eval_language}\n")
            continue
        except ValueError:
            print("No distances can be calculated, skipping this language")
            with open(
                f"./experiment_folder/logs/failed_languages_{task}{'_' + custom_args.output_name if custom_args.output_name else ''}.txt",
                "a",
            ) as f:
                f.write(f"{eval_language}\n")
            continue


if __name__ == "__main__":
    job_name = "clm_lang"
    debug = True
    partition = "a100"
    time = "01:00:00"

    master_dir = find_master()

    # Set the experi    ment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"
    pass_name = debug * "debug_" + job_name
    pass_partition = f"gpu_{partition}{debug * '_debug'}"
    run_count = update_submission_log(experiments_dir, pass_name)
    experiments_dir = experiments_dir / pass_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    # some shenanigans to pass a time argument through submitit
    first = sys.argv[1]
    if first.startswith("--"):
        job_input = sys.argv[1:]
    else:
        job_input = sys.argv[2:]
        if len(first) == 1:
            time = f"0{first}:00:00"
        else:
            job_name = job_name + f"_{first}"

    parameters = {
        "slurm_partition": pass_partition,
        "slurm_time": f"{'01:00:00' if pass_partition.endswith('debug') else time}",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": f"{'genius' if pass_partition.startswith(('gpu_p100', 'gpu_v100')) else 'wice'}",
            "account": os.environ["LAGOM_ACCOUNT"],  # we use lagom credits instead of "ACCOUNT_INFO"
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": f"{'' if pass_partition.endswith('debug') else 'stef.accou@student.kuleuven.be'}",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    # job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    print("job submitted")
