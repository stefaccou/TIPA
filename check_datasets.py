import submitit
import os
import sys
from cluster_submission_utils import find_master, update_submission_log


def main():
    from datasets import load_dataset
    import datasets

    sequence_target = 200_000
    langs = datasets.get_dataset_config_names("HuggingFaceFW/fineweb-2")

    langs_enough = []
    langs_not_enough = {}
    for lang in langs:
        print(f"Processing {lang}...")
        if lang == "eng_Latn":
            ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-2", lang, streaming=True, split="train")

        n_sentences = 0
        for sample in ds:
            for sent in sample["text"].split("\n"):
                if not (sent := sent.strip()):
                    continue
                n_sentences += 1
                if n_sentences == sequence_target:
                    break
            if n_sentences == sequence_target:
                break
        if n_sentences == sequence_target:
            langs_enough.append(lang)
        else:
            langs_not_enough[lang] = n_sentences
    print(f"languages with more than {sequence_target} sentences: {langs_enough}")
    print(f"languages with less than {sequence_target} sentences: {langs_not_enough}")


if __name__ == "__main__":
    job_name = "dataset_check"
    debug = True
    partition = "p100"
    time = "02:00:00"

    master_dir = find_master()

    # Set the experi    ment folder as a subdirectory of 'Master_thesis'
    experiments_dir = master_dir / "experiment_folder"
    pass_name = debug * "debug_" + job_name
    pass_partition = f"gpu_{partition}{debug * '_debug'}"
    run_count = update_submission_log(experiments_dir, pass_name)
    experiments_dir = experiments_dir / pass_name / f"{run_count:03d}"
    experiments_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    # some shenanigans to pass a time argument through submitit
    first = sys.argv[1]
    if first.startswith("--"):
        job_input = sys.argv[1:]
        pass_time = time
    else:
        job_input = sys.argv[2:]
        if len(first) == 1:
            pass_time = f"0{first}:00:00"
        else:
            pass_time = first
    parameters = {
        "slurm_partition": pass_partition,
        "slurm_time": f"{'01:00:00' if pass_partition.endswith('debug') else pass_time}",
        "slurm_job_name": job_name,
        "slurm_additional_parameters": {
            "clusters": f"{'genius' if pass_partition.startswith(('gpu_p100', 'gpu_v100')) else 'wice'}",
            "account": os.environ["LAGOM_ACCOUNT"],  # we use lagom credits instead of "ACCOUNT_INFO"
            "nodes": 1,
            "cpus_per_gpu": 16,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": f"{'' if pass_partition.endswith('debug') else 'stef.accou@student.kuleuven.be'}",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    # job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    print("job submitted")
