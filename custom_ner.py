import sys
import submitit
import os
from custom_submission_utils import find_master, update_submission_log


def main():
    from datasets import load_dataset
    from transformers import TrainingArguments, AutoTokenizer, AutoConfig
    from adapters import AdapterTrainer, AutoAdapterModel
    from adapters.composition import Stack
    from transformers import DataCollatorForTokenClassification
    import evaluate
    import numpy as np

    metric = evaluate.load("seqeval")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    raw_datasets = load_dataset("wikiann", "en")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[lab] for lab in label if lab != -100] for label in labels]
        true_predictions = [
            [label_names[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    id2label = {i: label for i, label in enumerate(label_names)}
    # label2id = {v: k for k, v in id2label.items()}

    """model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        id2label=id2label,
        label2id=label2id,
    )

    init(model)
    model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", load_as="en")
    model.add_adapter("ner")
    model.train_adapter(["ner"])
    model.active_adapters = Stack("en", "ner")"""
    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )
    model = AutoAdapterModel.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )
    # (Optionally) load language adapters if needed
    model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", load_as="en")
    model.add_adapter("ner")
    model.add_multiple_choice_head("ner", num_choices=len(id2label), id2label=id2label)
    model.train_adapter(["ner"])
    model.active_adapters = Stack("en", "ner")
    # print(model.active_adapters)

    training_args = TrainingArguments(
        output_dir="./trained_adapters/custom_ner_adapter",
        evaluation_strategy="epoch",
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
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    # we save the ner adapter as "ner_adapter"


if __name__ == "__main__":
    job_name = "better_ner_adapter_debug"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = "gpu_a100_debug"
    parameters = {
        "slurm_partition": partition,
        # "slurm_time": "03:00:00",
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else '10:30:00'}",
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
