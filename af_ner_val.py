# This script will go through all adapters in trained_adapters and return their score on Afrikaans NER
# We trained different language family adapters and configurations on CC100
# now it's time to evaluate down-stream performance

import sys
import submitit
from custom_submission_utils import find_master, update_submission_log


def main(FAMILY_ADAPTER):
    import numpy as np
    from transformers import EvalPrediction, TrainingArguments, AutoTokenizer
    from adapters import AutoAdapterModel, AdapterTrainer
    from adapters.composition import Stack
    from datasets import load_dataset

    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    dataset_af = load_dataset("wikiann", "af")
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
    model.load_adapter("/trained_adapters/ner")
    model.load_adapter(f"/trained_adapters/family/{FAMILY_ADAPTER}/mlm", load_as=FAMILY_ADAPTER)
    model.active_adapters = Stack(FAMILY_ADAPTER, "ner")
    print(model.active_adapters)

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./eval_output/{FAMILY_ADAPTER}",
            remove_unused_columns=False,
        ),
        eval_dataset=dataset_af,
        compute_metrics=compute_accuracy,
    )
    eval_trainer.evaluate()


if __name__ == "__main__":
    submitit_input = sys.argv[1:] if len(sys.argv) > 1 else "No input passed"
    job_name = f"af_ner_eval_{submitit_input}"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    parameters = {
        "slurm_partition": "gpu_p100_debug",
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

    job = executor.submit(main, submitit_input)
    print(f"{submitit_input} eval submitted")
