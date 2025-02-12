import submitit
import os

os.environ["SLURM_JOB_ID"] = "thisIsATest"
os.environ["SLURM_CLUSTER_NAME"] = "wice"


def add(a, b):
    return a + b


# SBATCH --job-name=test_for_slurm_and_stuff
# SBATCH --output=%x.out

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
# set timeout in min, and partition for running the job
executor.update_parameters(
    timeout_min=1,
    slurm_partition="wice",
    slurm_additional_parameters={
        "job_name": "Submitit_test",
        "clusters": "wice",
        "partition": "interactive",
        "account": "lp_hpcinfo",
        "nodes": 1,
        "cpus_per_gpu": 64,
        # "gpus_per_node": 1,
        "output": "/test_for_output",
    },
)
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
