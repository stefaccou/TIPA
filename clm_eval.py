import submitit
import sys
import os
from cluster_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        HfArgumentParser,
    )
    from datasets import load_dataset, Dataset, Split
    import adapters
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class DatasetArguments:
        """
        Arguments pertaining to what data we are going to import
        """

        # the list of languages to process, separated by spaces
        language: Optional[str] = field(
            default=None, metadata={"help": "List of languages to process, separated by comma"}
        )
        cache_dir: Optional[str] = field(
            default="data",
            metadata={"help": "Where to retrieve the downloaded datasets."},
        )
        adapter_dir: Optional[str] = field(
            default="outputs",
            metadata={"help": "Where to retrieve the trained model."},
        )

    parser = HfArgumentParser(DatasetArguments)

    # add a comma to refer to first part of tuple output
    (dataset_args,) = parser.parse_args_into_dataclasses(submit_arguments)

    model_name = "google/gemma-3-1b-pt"
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=dataset_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=dataset_args.cache_dir)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Gemma does this
        model.config.pad_token_id = tokenizer.pad_token_id
    # dataset = datasets.load_dataset(
    #     "openlanguagedata/flores_plus", dataset_args.language, cache_dir=dataset_args.cache_dir
    # )

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

    if dataset_args.language == "eng_Latn":
        ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-2", dataset_args.language, streaming=True, split="train")

    test_dataset = Dataset.from_generator(generate_lines, gen_kwargs={"dataset": ds, "cutoff": 2000}, split=Split.TEST)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            # no padding here; we'll do dynamic padding in the collator
        )

    tokenized_datasets = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        # remove_columns=dataset["devtest"].column_names,  # keep only model features
    )
    # We use a CLM data collator: pads dynamically and sets labels=input_ids with -100 on padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6) Evaluate
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
        do_eval=True,
        logging_dir="./logs",
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # eval_dataset=tokenized_datasets["devtest"],
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print("Evaluating on ", dataset_args.language)
    print("Evaluation without adapter")
    print(eval_results)
    adapters.init(model)

    model.load_adapter(dataset_args.adapter_dir, load_as="clm")
    model.set_active_adapters("clm")
    trainer = Trainer(
        model=model,
        args=training_args,
        # eval_dataset=tokenized_datasets["devtest"],
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    eval_results = trainer.evaluate()
    print("Evaluation with adapter")
    print(eval_results)


if __name__ == "__main__":
    job_name = "adapter_eval"
    debug = True
    partition = "p100"
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
        pass_time = time
    else:
        job_input = sys.argv[2:]
        if len(first) == 1:
            pass_time = f"0{first}:00:00"
        else:
            pass_time = first
    parameters = {
        "slurm_partition": pass_partition,
        "slurm_time": f"{'01:00:00' if pass_partition.endswith('debug') else pass_time}",
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
