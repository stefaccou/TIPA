# This script will go through all adapters in trained_adapters and return their score on Afrikaans NER
# We trained different language family adapters and configurations on CC100
# now it's time to evaluate down-stream performance

import sys
import submitit
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    import os
    import numpy as np
    from transformers import EvalPrediction, TrainingArguments, AutoTokenizer, HfArgumentParser
    from adapters import AutoAdapterModel, AdapterTrainer
    from adapters.composition import Stack
    from datasets import load_dataset
    from typing import Optional
    from dataclasses import dataclass, field
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    @dataclass
    class CustomArguments:
        """
        Arguments pertaining to what data we are going to input our model for training and eval.
        """

        # METHOD IMPLEMENTATION
        task_adapter_path: Optional[str] = field(
            default=None,
            metadata={
                "help": ("path for pre-trained task adapter"),
            },
        )
        language_adapter_path_template: Optional[str] = field(
            default=None,
            metadata={
                "help": ("path for pre-trained language adapter"),
            },
        )
        language_adapter_name: Optional[str] = field(
            default=None,
            metadata={
                "help": ("name of the language adapter"),
            },
        )

        def __post_init__(self):
            if self.language_adapter_path_template is None:
                raise ValueError("language_adapter_path_template must be provided")
            if self.language_adapter_name is None:
                raise ValueError("language_adapter_name must be provided")
            if self.task_adapter_path is None:
                raise ValueError("task_adapter_path must be provided")

    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    # we remove sys.argv as it interferes with parsing
    sys.argv = ""
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    if len(submit_arguments) == 1 and submit_arguments[0].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        custom_args, training_args = parser.parse_json_file(json_file=os.path.abspath(submit_arguments[0]))
    else:
        print("calling parser")
        custom_args, training_args = parser.parse_args_into_dataclasses(submit_arguments)
    print("custom args", custom_args)
    print(custom_args.language_adapter_name)
    print(custom_args.language_adapter_path_template)
    print(custom_args.task_adapter_path)

    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # If True, all tokens of a word will be labeled, otherwise only the first token
    label_all_tokens = True

    # Batch encoding function for NER
    def encode_batch(examples):
        # Tokenize word-level inputs
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        all_labels = []
        # Align word-level NER tags with tokenized inputs
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First token of the word
                    label_ids.append(labels[word_idx])
                else:
                    # Subsequent tokens
                    label_ids.append(labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    # Preprocessing helper
    def preprocess_dataset(dataset):
        dataset = dataset.map(encode_batch, batched=True)
        # Format for PyTorch
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    # Load and preprocess WikiANN (English)
    dataset_af = load_dataset("wikiann", "af", trust_remote_code=True)
    # Choose train split (or validation/test as needed)
    dataset_af = preprocess_dataset(dataset_af["validation"])

    # we load in the adapters
    # we print the current directory
    print(f"Current directory: {os.getcwd()}")
    model.load_adapter(adapter_name_or_path=custom_args.task_adapter_path, load_as="ner")
    model.load_adapter(
        custom_args.language_adapter_path_template.format(lang=custom_args.language_adapter_name),
        load_as=custom_args.language_adapter_name,
    )
    model.active_adapters = Stack(custom_args.language_adapter_name, "ner")
    print(model.active_adapters)

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    pdbs = training_args.per_device_eval_batch_size if training_args.per_device_eval_batch_size else 8
    eval_args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_eval_batch_size=pdbs,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        eval_accumulation_steps=2,
    )
    eval_trainer = AdapterTrainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset_af,
        compute_metrics=compute_accuracy,
    )
    eval_trainer.evaluate()


if __name__ == "__main__":
    # we want just the one argument as a string here

    job_name = "af_ner_eval"
    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_p100",
        "slurm_time": "00:15:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "genius",
            "account": "intro_vsc37220",  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "END, FAIL",
            "mail_user": "stef.accou@student.kuleuven.be",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)
    submitit_input = sys.argv[1:] if len(sys.argv) > 1 else "No input passed"
    job = executor.submit(main, submitit_input)
    print("eval submitted")
