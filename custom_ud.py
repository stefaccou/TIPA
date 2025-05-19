import sys
import submitit
import os
from custom_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from datasets import load_dataset
    from dataclasses import dataclass, field
    from transformers import (
        AutoModelForTokenClassification,
        TrainingArguments,
        XLMRobertaTokenizerFast,
        HfArgumentParser,
        EarlyStoppingCallback,
    )
    from adapters import AdapterTrainer, init
    from adapters.composition import Stack
    from transformers import DataCollatorForTokenClassification
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    raw_datasets = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    label_names = raw_datasets["train"].features["upos"].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

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

    # apply to train/validation
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
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

    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        id2label=id2label,
        label2id=label2id,
    )
    init(model)
    print("initted model")
    model.load_adapter("./trained_adapters/en", load_as="en")
    model.add_adapter("ud_pos")
    model.train_adapter(["ud_pos"])
    model.active_adapters = Stack("en", "ud_pos")
    print(model.active_adapters)

    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        weight_decay=0.01,
        save_steps=25000,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )
    trainer = AdapterTrainer
    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,  # future versions will accept `processing_class` instead
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    # we save the ner adapter as "ner_adapter"
    trainer.train()
    model.save_adapter("ud_pos", data_args.output_dir)


if __name__ == "__main__":
    debug = True
    job_name = debug * "debug_" + "convergence_pos_adapter"

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
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else '05:30:00'}",
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
