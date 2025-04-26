import sys
import submitit
import os
from custom_submission_utils import find_master, update_submission_log


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import TrainingArguments, AutoTokenizer, AutoConfig, EvalPrediction, AutoModelForMultipleChoice
    from adapters import AdapterTrainer, init, AdapterConfig
    from adapters.composition import Stack
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    raw_datasets = load_dataset("super_glue", "copa")

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

    tokenized_datasets = preprocess_dataset(raw_datasets)
    # Load the model configuration and adapter model
    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
        num_labels=2,  # 2 choices per example
        id2label={0: "choice1", 1: "choice2"},
        label2id={"choice1": 0, "choice2": 1},
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )
    init(model)
    # (Optionally) load language adapters if needed
    model.load_adapter("./trained_adapters/en", load_as="en")
    model.add_adapter("copa")
    """model.add_multiple_choice_head(
        "copa", num_choices=2, id2label={0: "choice1", 1: "choice2"}
    )"""
    model.train_adapter(["copa"])
    model.active_adapters = Stack("en", "copa")
    print(model.active_adapters)

    training_args = TrainingArguments(
        output_dir="./trained_adapters/custom_copa_adapter",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=250,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )
    train_dataset = concatenate_datasets([tokenized_datasets["train"], tokenized_datasets["validation"]])
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    # we save the ner adapter as "ner_adapter"


if __name__ == "__main__":
    # we want just the one argument as a string here

    job_name = "training_custom_copa_adapter"
    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_p100",
        "slurm_time": "02:00:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "genius",
            "account": os.environ["ACCOUNT_INFO"],  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "END",
            "mail_user": "",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)
    submitit_input = sys.argv[1:] if len(sys.argv) > 1 else "No input passed"
    job = executor.submit(main)
    print("eval submitted")
