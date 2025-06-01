import sys
import submitit
import os
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from datasets import load_dataset
    from dataclasses import dataclass, field
    from transformers import (
        TrainingArguments,
        AutoTokenizer,
        HfArgumentParser,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        AutoModelForSequenceClassification,
        Trainer,
    )
    from transformers.trainer_utils import get_last_checkpoint
    import evaluate
    import numpy as np

    metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer)

    @dataclass
    class DataTrainingArguments:
        """
        Arguments pertaining to what data we are going to input our model for training and eval.
        """

        output_dir: str = field(
            default=None,
            metadata={"help": ("Directory for model checkpoints and saving")},
        )

    parser = HfArgumentParser(DataTrainingArguments)
    sys.argv = ""
    if len(submit_arguments) == 1 and submit_arguments[0].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(submit_arguments[0]))
    else:
        print("calling parser")
        # add a comma to refer to first part of tuple output
        (data_args,) = parser.parse_args_into_dataclasses(submit_arguments)

    print("passed args: ", data_args)

    raw_datasets = load_dataset("Davlan/sib200", "eng_Latn")
    label_list = [
        "science/technology",
        "travel",
        "politics",
        "sports",
        "health",
        "entertainment",
        "geography",
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # From Huggingface question answering example, adapted slightly for evaluation
    def preprocess_function(examples):
        # Tokenize the "text" field; returns a dict with 'input_ids', 'attention_mask', etc.
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,  # change max_length if you need longer/shorter
        )
        # Map each category string to its integer ID
        # (when using batched=True, examples["category"] is a list of strings)
        tokens["labels"] = [label2id[cat] for cat in examples["category"]]
        return tokens

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # load in model and adapters
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1000,
        weight_decay=0.01,
        overwrite_output_dir=False,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    last_checkpoint = get_last_checkpoint(data_args.output_dir)
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        training_args.resume_from_checkpoint = last_checkpoint
    else:
        print("No checkpoint found, starting training from scratch.")
    trainer.train()
    # we save the qa finetune as "qa"
    # we make a folder output_dir + / "final" to save the model
    final_output_dir = data_args.output_dir + "/final"
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    trainer.save_model(final_output_dir)


if __name__ == "__main__":
    debug = False
    job_name = "debug_" * debug + "convergence_sib_finetune"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = f"gpu_a100{'_debug' * debug}"
    parameters = {
        "slurm_partition": partition,
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else '5:30:00'}",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": f"{'genius' if partition.startswith('gpu_p100') else 'wice'}",
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
