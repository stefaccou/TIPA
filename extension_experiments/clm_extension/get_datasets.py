import submitit
import sys
import os
from cluster_submission_utils import find_master, update_submission_log


def main(submit_arguments):
    from datasets import load_dataset, Dataset, Split
    from dataclasses import dataclass, field
    from typing import Optional
    from transformers import HfArgumentParser

    @dataclass
    class DatasetArguments:
        """
        Arguments pertaining to what data we are going to import
        """

        # the list of languages to process, separated by spaces
        languages: Optional[str] = field(
            default=None, metadata={"help": "List of languages to process, separated by comma"}
        )
        cache_dir: Optional[str] = field(
            default="data",
            metadata={"help": "Where to store the downloaded datasets."},
        )

    parser = HfArgumentParser(DatasetArguments)

    # add a comma to refer to first part of tuple output
    (dataset_args,) = parser.parse_args_into_dataclasses(submit_arguments)
    if dataset_args.languages:
        dataset_args.languages = [lang.strip() for lang in dataset_args.languages.split(",")]
    else:
        dataset_args.languages = []

    langs = {
        "Thai": ["tha_Thai"],
        "Myanmar (Burmese)": ["mya_Mymr"],
        "Hindi": ["hin_Deva", "hin_Latn"],
        "Ilocano": ["ilo_Latn"],
        "Haitian Creole": ["hat_Latn"],
        "Turkish": ["tur_Latn"],
        "Maori": ["mri_Latn"],
        "Vietnamese": ["vie_Latn"],
        "Icelandic": ["isl_Latn"],
        "Italian": ["ita_Latn"],
        "Tamil": ["tam_Latn", "tam_Taml"],
        "Javanese": ["jav_Latn"],
        "Japanese": ["jpn_Jpan"],
        "German": ["deu_Latn"],
        "Greek": ["ell_Grek"],
        "Russian": ["rus_Cyrl"],
        "Indonesian": ["ind_Latn"],
        "Spanish": ["spa_Latn"],
        "Turkmen": ["tuk_Arab", "tuk_Cyrl", "tuk_Latn"],
        "Eastern Mari": ["mhr_Cyrl"],
        "Mingrelian": ["xmf_Geor"],
        "Basque": ["eus_Latn"],
        "Serbian": ["srp_Cyrl", "srp_Latn"],
        "Chinese": ["cmn_Hani"],
        "Arabic": ["arb_Arab"],
        "Estonian": ["ekk_Latn"],
        "Swahili": ["swh_Latn"],
        "English": ["eng_Latn"],
    }

    # we check which languages were passed as submit_arguments
    to_check = {}
    if not dataset_args.languages:
        to_check = langs
    else:
        for lang in dataset_args.languages:
            if lang in langs:
                to_check[lang] = langs[lang]
            else:
                print(f"Language {lang} not recognized, skipping.")

    def generate_lines(dataset, cutoff):
        n = 0
        for example in dataset:
            if n >= cutoff:
                break
            for line in example["text"].split("\n"):
                if good_line := line.strip():
                    n += 1
                    # print(n, end="\r")
                    yield {"text": good_line}

    sequence_target = 200_000
    langs_enough = []
    langs_not_enough = {}
    for language in to_check.keys():
        print(f"Processing {language}...")
        for lang in to_check[language]:
            # we first check if data/fineweb/lang path exists
            if os.path.exists(f"{dataset_args.cache_dir}/{lang}"):
                print(f"Skipping {lang}, already exists")
                continue
            if lang == "eng_Latn":
                ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
            else:
                ds = load_dataset("HuggingFaceFW/fineweb-2", lang, streaming=True, split="train")

            n_sentences = 0
            try:
                for sample in ds:
                    for sent in sample["text"].split("\n"):
                        if not (sent := sent.strip()):
                            continue
                        n_sentences += 1
                        # print(n_sentences, end="\r")
                        if n_sentences == sequence_target:
                            break
                    if n_sentences == sequence_target:
                        break
            except Exception as e:
                print(f"Error processing {lang}: {e}")
                n_sentences = 0
            if n_sentences == sequence_target:
                langs_enough.append(lang)
                new_dataset = Dataset.from_generator(
                    generate_lines, gen_kwargs={"dataset": ds, "cutoff": 200000}, split=Split.TRAIN
                )
                new_dataset.save_to_disk(f"{dataset_args.cache_dir}/{lang}")
            else:
                print("not enough for ", lang, n_sentences)
                langs_not_enough[lang] = n_sentences


if __name__ == "__main__":
    job_name = "dataset_processing"
    debug = True
    partition = "p100"
    time = "06:00:00"

    master_dir = find_master()

    # Set the experi    ment folder as a subdirectory of 'TIPA'
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
            "mail_user": f"{'' if pass_partition.endswith('debug') else os.environ["MAIL_ADDRESS"]}",
        },
    }

    # Initialize the Submitit executor with the new experiments_dir
    executor = submitit.AutoExecutor(folder=str(experiments_dir))
    executor.update_parameters(**parameters)

    # job_input = sys.argv[1:] if len(sys.argv) > 1 else "default text"

    job = executor.submit(main, job_input)
    print("job submitted")
