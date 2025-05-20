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
        load_finetuned_model,
    )

    from transformers import (
        AutoTokenizer,
        HfArgumentParser,
        XLMRobertaTokenizerFast,
        DataCollatorForTokenClassification,
        DefaultDataCollator,
        Trainer,
    )

    import os
    import gc
    import json
    import torch
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

        def __post_init__(self):
            # we check if distance is in the list of available distances
            # OR a list combination of these types

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

    task = custom_args.task

    eval_languages = get_eval_languages(task)
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
    data_collator = (
        DataCollatorForTokenClassification(tokenizer=tokenizer) if not task == "qa" else DefaultDataCollator()
    )
    model = load_finetuned_model(task)

    if not custom_args.limit:
        limit_str = ""
    else:
        if custom_args.limit < 1:
            decimal = str(custom_args.limit).split(".")[1]
            limit_str = f"_0{decimal}"
        else:
            custom_args.limit = int(custom_args.limit)
            limit_str = f"_{str(custom_args.limit)}"

    print(f"\n{'~' * 30}\n{task.upper()}\n{'~' * 30}")
    # Brute: we only consider english for now
    for eval_language in eval_languages.keys():
        try:
            print(
                "\n\n",
                f"Evaluating {task} on {eval_language} ({ld.get(eval_language, tag_type=TagType.BCP_47_CODE).english_name})",
            )

            # Load and preprocess the dataset
            dataset_eval = load_eval(task, eval_language, eval_languages)
            tokenized_datasets = preprocess(dataset_eval, task, tokenizer)

            if task == "ner":
                ner_feature = dataset_eval.features["ner_tags"]
                label_names = ner_feature.feature.names
            else:
                label_names = None

            # dataset is now ready to be used with adapters for cross-lingual transfer

            def run_eval(model, name):
                # we load in the task adapter

                compute_metrics = get_compute_metrics(task, label_names)
                trainer_kwargs = get_trainer_kwargs(
                    task, model, tokenized_datasets, tokenizer, data_collator, compute_metrics
                )
                # instantiate
                eval_trainer = Trainer(**trainer_kwargs)
                if not task == "qa":
                    ev = eval_trainer.evaluate()
                else:
                    predictions, _, _ = eval_trainer.predict(tokenized_datasets)
                    start_logits, end_logits = predictions
                    ev = compute_metrics(start_logits, end_logits, tokenized_datasets, dataset_eval)

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
            # we evaluate the model
            evaluations["finetuned_model"] = run_eval(model, "finetuned_model")

            if not os.path.exists(f"./finetuned_models/results/{eval_language}"):
                os.makedirs(f"./finetuned_models/results/{eval_language}")
            # we save this
            if custom_args.output_name:
                output_file = (
                    f"./finetuned_models/results/{eval_language}/{task}_{custom_args.output_name}{limit_str}.json"
                )
            else:
                output_file = f"./finetuned_models/results/{eval_language}/{task}_eval{limit_str}.json"
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
    debug = False
    job_name = debug * "debug_" + "convergence_finetune_task"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = f"gpu_p100{debug * '_debug'}"
    parameters = {
        "slurm_partition": partition,
        # "slurm_time": "03:00:00",
        "slurm_time": f"{'00:30:00' if partition.endswith('debug') else '00:30:00'}",
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

    job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    # job = executor.submit(main)
    print("job submitted")
