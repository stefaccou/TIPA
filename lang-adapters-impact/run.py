import submitit


def main():
    executor = submitit.AutoExecutor(folder="experiment_folder")
    executor.update_parameters(
        slurm_partition="gpu_a100_debug",
        slurm_time="00:10:00",
        slurm_job_name="adapters_test_run",
        slurm_additional_parameters={
            "clusters": "wice",
            "account": "intro_vsc37220",
            "nodes": 1,
            "cpus_per_gpu": 64,
            "gpus_per_node": 1,
            "mail_type": "BEGIN,END,FAIL",
            "mail_user": "",
        },
    )

    command = [
        "python",
        "lang-adapters-impact/adapters.py",
        "--task",
        "xnli",
        "--data_setup",
        "all",
        "--train_lang",
        "en",
        "--base_model",
        "xlm-roberta-base",
        "--seed",
        "1",
        "--epochs",
        "3",
        "--train_output_dir",
        "results/training",
        "--eval_output_dir",
        "results/eval",
    ]

    job = executor.submit(submitit.helpers.CommandFunction(command))
    job.result()


if __name__ == "__main__":
    main()
