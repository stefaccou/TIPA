import submitit
import os


def run_adapters_script():
    # Define the command to run the adapters.py script with the specified arguments
    command = (
        "python adapters.py "
        "--task 'xnli' "
        "--data_setup 'all' "
        "--train_lang 'en' "
        "--base_model 'xlm-roberta-base' "
        "--seed 1 "
        "--epochs 3 "
        "--train_output_dir 'results/training' "
        "--eval_output_dir 'results/eval'"
    )

    # Execute the command
    os.system(command)


# Set up the submitit executor
executor = submitit.AutoExecutor(folder="submitit_logs")

parameters = {
    "slurm_partition": "gpu_a100_debug",
    "slurm_time": "00:10:00",
    "slurm_job_name": "test_debugging",  # Put something informative here, it helps tracking down issues easier
    "slurm_additional_parameters": {
        "clusters": "wice",
        "account": "intro_vsc37220",  # Your VSC account
        "nodes": 1,
        "cpus_per_gpu": 64,
        "gpus_per_node": 1,
        "mail_type": "BEGIN,END,FAIL",
        "mail_user": "",  # for notifications specified above
    },
}

# Update the parameters
executor.update_parameters(**parameters)

# Submit the job
job = executor.submit(run_adapters_script)

# Print the job ID
print(f"Submitted job with ID: {job.job_id}")
