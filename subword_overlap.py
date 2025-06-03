import submitit
import os
import sys
from cluster_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from unseen_eval import (
        get_eval_languages,
        load_eval,
    )
    from datasets import load_dataset, concatenate_datasets

    from transformers import (
        AutoTokenizer,
        HfArgumentParser,
        XLMRobertaTokenizerFast,
    )

    import os
    import json
    from qq import LanguageData, TagType

    from dataclasses import dataclass, field
    from typing import Optional
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

        output_name: Optional[str] = field(
            default=None,
            metadata={
                "help": (
                    "The name of the output file. "
                    "If not specified, it will be saved as task_eval.json in the trained_adapters folder."
                )
            },
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

    Tokenizer = XLMRobertaTokenizerFast if task == "pos" else AutoTokenizer
    tokenizer = Tokenizer.from_pretrained("xlm-roberta-base")

    if task == "ner":
        source_corpus = load_dataset("wikiann", "en")
        tokenized_corpus = source_corpus["train"].map(
            lambda x: tokenizer(x["tokens"], is_split_into_words=True, truncation=True),
            batched=True,
        )

    elif task == "pos":
        source_corpus = load_dataset("universal_dependencies", "en_ewt")
        tokenized_corpus = source_corpus["train"].map(
            lambda x: tokenizer(x["tokens"], is_split_into_words=True, truncation=True),
            batched=True,
        )
    elif task == "qa":
        source_corpus = load_dataset("squad")
        # we print the features of the dataset
        tokenized_corpus = source_corpus["train"].map(
            lambda x: tokenizer(
                x["context"],
                x["question"],
                truncation=True,
            ),
            batched=True,
        )
    elif task == "copa":
        source_corpus = load_dataset("super_glue", "copa")
        tokenized_corpus = source_corpus["train"].map(
            lambda x: tokenizer(x["premise"], x["question"], x["choice1"], x["choice2"], truncation=True),
            batched=True,
        )
        tokenized_eval = source_corpus["validation"].map(
            lambda x: tokenizer(x["premise"], x["question"], x["choice1"], x["choice2"], truncation=True),
            batched=True,
        )
        # we concatenate the training and validation sets
        tokenized_corpus = concatenate_datasets([tokenized_corpus, tokenized_eval])
    elif task == "sib":
        source_corpus = load_dataset("Davlan/sib200", "eng_Latn")
        tokenized_corpus = source_corpus["train"].map(
            lambda x: tokenizer(x["text"], truncation=True),
            batched=True,
        )

    # we make a set of all the tokens in the source corpus
    train_vocab = set()
    for example in tokenized_corpus:
        train_vocab.update(example["input_ids"])
    # we print the first few

    token_overlaps = {}
    for eval_language in eval_languages.keys():
        if task == "sib":
            eval_lang, script = eval_language
            eval_lang = eval_lang + "_" + script
        else:
            eval_lang = eval_language
            script = None

        try:
            print(
                "\n\n",
                f"Evaluating token overlap for {task} on {eval_lang} ({ld.get(eval_lang, tag_type=TagType.BCP_47_CODE).english_name})",
            )
        except KeyError:
            print(
                "\n\n",
                f"Evaluating token overlap for {task} on {eval_lang} (no ld data)",
            )

        # Load and preprocess the dataset
        dataset_eval = load_eval(task, eval_language, eval_languages)
        # we do the same here
        if task == "ner":
            tokenized_eval = dataset_eval.map(
                lambda x: tokenizer(x["tokens"], is_split_into_words=True, truncation=True),
                batched=True,
            )
        elif task == "pos":
            tokenized_eval = dataset_eval.map(
                lambda x: tokenizer(x["tokens"], is_split_into_words=True, truncation=True),
                batched=True,
            )
        elif task == "qa":
            tokenized_eval = dataset_eval.map(
                lambda x: tokenizer(
                    x["context"],
                    x["question"],
                    truncation=True,
                ),
                batched=True,
            )
        elif task == "copa":
            tokenized_eval = dataset_eval.map(
                lambda x: tokenizer(x["premise"], x["question"], x["choice1"], x["choice2"], truncation=True),
                batched=True,
            )
        elif task == "sib":
            tokenized_eval = dataset_eval.map(
                lambda x: tokenizer(x["text"], truncation=True),
                batched=True,
            )

        test_tokens = [token for example in tokenized_eval for token in example["input_ids"]]
        total_test_tokens = len(test_tokens)

        # token-level coverage
        if total_test_tokens > 0:
            overlap_count = sum(1 for token in test_tokens if token in train_vocab)
            token_coverage = overlap_count / total_test_tokens
        else:
            overlap_count = 0
            token_coverage = 0.0

        # calculate type-level coverage
        test_vocab = set(test_tokens)
        if test_vocab:
            type_overlap_count = len(test_vocab.intersection(train_vocab))
            type_overlap = type_overlap_count / len(test_vocab)
        else:
            type_overlap_count = 0
            type_overlap = 0.0

        token_overlaps[eval_lang] = {
            "token_coverage": token_coverage,
            "overlap_count": overlap_count,
            "type_overlap": type_overlap,
            "type_overlap_count": type_overlap_count,
        }
        print(f"Token coverage: {token_coverage:.2%} ({overlap_count} of {total_test_tokens} tokens)")
        print(f"Type-level overlap: {type_overlap:.2%} ({type_overlap_count} unique tokens)")
        if task == "ner":
            train_hits = []
            for example in tokenized_corpus:
                for token, label in zip(example["input_ids"], example["ner_tags"]):
                    if token in test_vocab:
                        train_hits.append(label)
            # we calculate the overlap
            entity_overlap = len(set(train_hits)) / len(test_vocab) if test_vocab else 0
            token_overlaps[eval_lang]["entity_overlap"] = entity_overlap
    # Once we have all, we save the results
    output_dir = os.path.join("./eval_output", "token_overlap")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{task}_token_overlap.json")
    with open(output_file, "w") as f:
        json.dump(token_overlaps, f, indent=4)
    print(f"Token overlap results saved to {output_file}")


if __name__ == "__main__":
    debug = False
    job_name = debug * "debug_" + "token_overlap"

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
