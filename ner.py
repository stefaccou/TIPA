import submitit
import os
from custom_submission_utils import find_master, update_submission_log


def main():
    from adapters import AutoAdapterModel, AdapterTrainer
    from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
    from datasets import load_dataset
    from custom_submission_utils import CustomTokenClassificationHead

    # Load the CoNLL-2003 dataset for NER
    dataset = load_dataset("wikiann", "en")
    label_list = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)

    # Load a pre-trained tokenizer (using a RoBERTa-based model)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)

    # Function to tokenize texts and align labels with tokens
    def tokenize_and_align_labels(examples, label_all_tokens=True):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,  # adjust max_length as needed
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id of None
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word, unless label_all_tokens is set to True
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    # Preprocess the dataset using the tokenization function above
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load the model configuration and adapter model
    config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=num_labels)
    model = AutoAdapterModel.from_pretrained("xlm-roberta-base", config=config)

    # (Optionally) load language adapters if needed
    model.load_adapter("trained_adapters/en")

    # Add a new task adapter for NER and a token classification head
    model.add_adapter("ner")
    # we cannot use the classic add_token_classification_head with non-BERT-based models
    # model.add_classification_head("ner", num_labels=num_labels)
    # Instead of using add_classification_head, manually attach a token classification head
    model.token_classification_head = CustomTokenClassificationHead(config)

    # Activate training for the NER adapter
    model.train_adapter(["ner"])

    # (The remainder of the script should include training arguments and trainer initialization)
    # For example, using the Transformers Trainer:
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="trained_adapters/ner",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="experiment_folder",
        logging_steps=5,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    # Initialize the AdapterTrainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Train the model using the Trainer
    trainer.train()


if __name__ == "__main__":
    job_name = "ner_training"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_a100",
        "slurm_time": "01:15:00",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": os.environ["ACCOUNT_INFO"],  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    # job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    # job = executor.submit(main, job_input)
    job = executor.submit(main)
    print("job submitted")
