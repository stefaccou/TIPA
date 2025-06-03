import sys
import submitit
import os
from cluster_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    import collections
    from tqdm import tqdm
    import numpy as np
    import evaluate
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        HfArgumentParser,
        TrainingArguments,
        DefaultDataCollator,
    )
    from adapters import Stack, AutoAdapterModel, AdapterTrainer
    from dataclasses import dataclass, field

    squad = load_dataset("squad")

    @dataclass
    class DataTrainingArguments:
        """
        Arguments pertaining to what data we are going to input our model for training and eval.
        """

        output_dir: str = field(
            default=None,
            metadata={"help": ("Directory for model checkpoints and saving")},
        )
        qa_adapter: str = field(
            default=None,
            metadata={"help": ("Path to the trained QA adapter")},
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

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", load_as="en")
    model.load_adapter(data_args.qa_adapter, load_as="qa")
    model.active_adapters = Stack("en", "qa")

    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

        inputs["example_id"] = example_ids
        return inputs

    validation_dataset = squad["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=squad["validation"].column_names,
    )
    data_collator = DefaultDataCollator()

    args = TrainingArguments(
        output_dir=data_args.output_dir,
        fp16=True,
    )
    trainer = AdapterTrainer(
        model=model,
        args=args,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    metrics = compute_metrics(start_logits, end_logits, validation_dataset, squad["validation"])
    print(metrics)


if __name__ == "__main__":
    debug = True
    job_name = f"{'debug_' * debug}qa_eval"

    master_dir = find_master()

    # Set the experiment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"

    run_count = update_submission_log(experiments_dir, job_name)
    experiments_dir = experiments_dir / job_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    partition = f"gpu_p100{'_debug' * debug}"
    parameters = {
        "slurm_partition": partition,
        # "slurm_time": "03:00:00",
        "slurm_time": f"{'01:00:00' if partition.endswith('debug') else '5:00:00'}",
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
