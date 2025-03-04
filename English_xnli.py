import torch
import adapters.composition as composition
from adapters import AutoAdapterModel, AdapterTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments
import submitit

# we will use XLM Roberta


def main():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    # load the dataset
    dataset_en = load_dataset("xnli", "en")

    def encode_batch(batch):
        all_encoded = {"input_ids": [], "attention_mask": []}
        for i in range(len(batch["premise"])):
            encoded = tokenizer(batch["premise"][i], batch["hypothesis"][i], padding="max_length", truncation=True)
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["input_ids"] = torch.tensor(all_encoded["input_ids"])
        all_encoded["attention_mask"] = torch.tensor(all_encoded["attention_mask"])
        all_encoded["label"] = torch.tensor(batch["label"])
        return all_encoded

    def preprocess_dataset(dataset):
        dataset = dataset.map(encode_batch, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    dataset_eng = preprocess_dataset(dataset_en)

    model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer")
    model.add_adapter("xnli_adapter")
    model.add_classification_head("xnli_adapter", num_labels=3)
    model.train_adapter(["xnli_adapter"])
    model.active_adapters = composition.Stack("en", "xnli_adapter")

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=10,
        output_dir="$VSC_DATA/xnli_test/training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_eng["train"],
    )
    trainer.train()



if __name__ == "__main__":
    parameters = {
        "slurm_partition": "gpu_a100",
        "slurm_time": "03:00:00",
        "slurm_job_name": "training xnli adapter for english",
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": "intro_vsc37220",
            "nodes": 1,
            "cpus_per_gpu": 18,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "stef.accou@student.kuleuven.be",
        },
    }

    executor = submitit.AutoExecutor(folder="xnli_test")
    executor.update_parameters(**parameters)

    job = executor.submit(main)
    job.result()
