from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import adapters
from adapters import AutoAdapterModel
from datasets import load_dataset, Dataset
import pandas as pd
import submitit
import os
import math
import pprint
import evaluate



def main():
    VSC_DATA = os.environ["VSC_DATA"]
    print(VSC_DATA)
    VSC_SCRATCH = os.environ["VSC_SCRATCH"]
    EVAL_FILE = VSC_DATA + "/Data/nl_val.txt"
    checkpoints = [
        VSC_SCRATCH+"/test-mlm/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-19000/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-500/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-17500/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-26000/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-14500/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-21000/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-5500/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-4000/mlm",
        VSC_SCRATCH+"/test-mlm/checkpoint-22000/mlm",
    ]

    # Load validation set
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    #dataset = load_dataset("$VSC_DATA/Data/nl_val.txt")
    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    adapters.init(model)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Strip and wrap into a DataFrame
    data = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(data)
    eval_dataset = dataset.select(range(500))
    print("Tokenizing dataset")
    # Tokenize
    tokenized_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128),
        batched=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    results = {}
    print("starting checkpoint loop")
    for checkpoint_path in checkpoints:
        print(f"Evaluating {checkpoint_path}...")
        adapter = model.load_adapter(checkpoint_path)
        model.set_active_adapters(adapter)

        training_args = TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=8,
            do_train=False,
            do_eval=True,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        eval_result = trainer.evaluate()
        pprint.pprint(eval_result)
        try:
            perplexity = math.exp(eval_result["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        results[checkpoint_path] = perplexity
        print(f"Perplexity for {checkpoint_path}: {perplexity}")

    # Find best model
    best_checkpoint = min(results, key=results.get)
    print(f"\nBest checkpoint: {best_checkpoint} with loss {results[best_checkpoint]}")


if __name__ == "__main__":
    parameters = {
        "slurm_partition": "gpu_a100_debug",
        "slurm_time": "00:10:00",
        "slurm_job_name": "adapter evaluation final debug",
        "slurm_additional_parameters": {
            "clusters": "wice",
            "account": "intro_vsc37220",  # replace with your account
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "",
        },
    }

    executor = submitit.AutoExecutor(folder="experiment_folder/eval")
    executor.update_parameters(**parameters)
    job = executor.submit(main)
