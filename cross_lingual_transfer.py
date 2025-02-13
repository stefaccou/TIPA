import submitit
import numpy as np
import torch
from adapters import AdapterTrainer, AutoAdapterModel
from adapters.composition import Stack
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoConfig, EvalPrediction, TrainingArguments


def main():
    # Make sure we use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_en = load_dataset("super_glue", "copa", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def encode_batch(examples):
        """Encodes a batch of input data using the model tokenizer."""
        all_encoded = {"input_ids": [], "attention_mask": []}
        for premise, question, choice1, choice2 in zip(
            examples["premise"], examples["question"], examples["choice1"], examples["choice2"]
        ):
            sentences_a = [premise + " " + question for _ in range(2)]
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
        dataset = dataset.map(encode_batch, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
        return dataset

    dataset_en = preprocess_dataset(dataset_en)

    config = AutoConfig.from_pretrained("xlm-roberta-base")
    model = AutoAdapterModel.from_pretrained("xlm-roberta-base", config=config)

    adap1 = model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer")
    adap2 = model.load_adapter("AdapterHub/xlm-roberta-base-zh-wiki_pfeiffer")
    model.set_active_adapters([adap1, adap2])

    model.add_adapter("copa")
    model.add_multiple_choice_head("copa", num_choices=2)
    model.train_adapter(["copa"])
    model.active_adapters = Stack("en", "copa")

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=100,
        output_dir="./training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )

    train_dataset = concatenate_datasets([dataset_en["train"], dataset_en["validation"]])

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    dataset_zh = load_dataset("xcopa", "zh", trust_remote_code=True)
    dataset_zh = preprocess_dataset(dataset_zh)

    model.active_adapters = Stack("zh", "copa")

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_output",
            remove_unused_columns=False,
        ),
        eval_dataset=dataset_zh["test"],
        compute_metrics=compute_accuracy,
    )
    eval_trainer.evaluate()

    # Save all adapters
    for adapter_name in model.config.adapters.adapters:
        model.save_adapter(adapter_name, f"{adapter_name}_saved")


if __name__ == "__main__":
    parameters = {
        "slurm_partition": "gpu_a100_debug",
        "slurm_time": "00:10:00",
        "slurm_job_name": "cross_lingual_transfer",
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": "intro_vsc37220",
            "nodes": 1,
            "cpus_per_gpu": 64,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "",
        },
    }

    executor = submitit.AutoExecutor(folder="experiment_folder")
    executor.update_parameters(**parameters)

    job = executor.submit(main)
    job.result()
