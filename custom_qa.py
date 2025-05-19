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
        AutoModelForQuestionAnswering,
        HfArgumentParser,
        EvalPrediction,
        EarlyStoppingCallback,
    )
    from adapters import AdapterTrainer, init
    from adapters.composition import Stack
    from transformers import DefaultDataCollator
    import evaluate
    import collections
    import numpy as np
    from tqdm import tqdm

    metric = evaluate.load("squad")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    raw_datasets = load_dataset("squad")
    data_collator = DefaultDataCollator()

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

    # From Huggingface question answering example
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def compute_metrics(eval_pred: EvalPrediction, features, examples, n_best=20):
        start_logits, end_logits = eval_pred.predictions
        example_to_features = collections.defaultdict(list)
        for i, f in enumerate(features):
            example_to_features[f["example_id"]].append(i)

        predicted_answers = []
        for example in tqdm(examples):
            qid = example["id"]
            context = example["context"]
            feature_indices = example_to_features[qid]
            best_answer = ""
            best_score = -float("inf")

            # Iterate over features for this example
            for idx in feature_indices:
                start_logit = start_logits[idx]
                end_logit = end_logits[idx]
                offsets = features[idx]["offset_mapping"]

                # Select top n_best start/end indices
                start_idxs = np.argsort(start_logit)[-n_best:]
                end_idxs = np.argsort(end_logit)[-n_best:]
                for si in start_idxs:
                    for ei in end_idxs:
                        if si <= ei and offsets[si] and offsets[ei]:
                            score = start_logit[si] + end_logit[ei]
                            if score > best_score:
                                best_score = score
                                best_answer = context[offsets[si][0] : offsets[ei][1]]

            if best_answer == "":
                best_answer = ""  # No valid span found
            predicted_answers.append({"id": qid, "prediction_text": best_answer})

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        results = metric.compute(predictions=predicted_answers, references=references)
        return {"exact_match": results["exact_match"], "f1": results["f1"]}

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")

    init(model)
    model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", load_as="en")
    model.add_adapter("qa")
    model.train_adapter(["qa"])
    model.active_adapters = Stack("en", "qa")

    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=25000,
        num_train_epochs=100,
        weight_decay=0.01,
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
        compute_metrics=lambda p: compute_metrics(p, tokenized_datasets["validation"], raw_datasets["validation"]),
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        processing_class=tokenizer,
    )
    trainer.train()
    # we save the qa adapter as "qa_adapter"
    model.save_adapter(data_args.output_dir, "qa_adapter")


if __name__ == "__main__":
    debug = True
    job_name = "debug_" * debug + "convergence_qa_adapter"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = f"gpu_p100{'_debug' * debug}"
    parameters = {
        "slurm_partition": partition,
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else '10:00:00'}",
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
