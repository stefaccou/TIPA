import submitit


def expensive_function(a, b):
    return a + b


# Either define 'manually' or read in from yaml or Hydra:
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

# All logs and slurm files will be stored in this folder
executor = submitit.AutoExecutor(folder="experiment_folder")
executor.update_parameters(**parameters)

job = executor.submit(expensive_function, 5, 7)
# This waits for completion and returns output, often not a good idea with long jobs,
# then you can just let this script finish.
output = job.result()
assert output == 12  # computed in the cluster
