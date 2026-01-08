# TIPA: Typologically Informed Parameter Aggregation

TIPA is a package for typologically informed parameter aggregation.
It provides tools to combine any number of architecturally identical modules, with a customisable weighting function.
We also provide a weighting function based on typological distance between a source and target language for zero-shot cross-lingual transfer.

For more info, we refer to notebook "0_Method_implementation.ipynb" in the repository,
or to our paper [TBD].

+ The files for creating the task adapters or English-language finetunings can be found in directory `adapters_finetunings`.
+ Analysis files, as well as additional figures and tables, can be found in directory `analysis`.
+ The averaged adapter used in the baseline calculations is joined under `trained_adapters`.
+ All scores for the full set of evaluations can be found under `eval_scores`.
+ Additional (unfinished) extension experiments are added to folder `extension-experiments`

The actual experiments were conducted in files `1_tipa_adapters.py` and finetune baselines calculated in `2_finetune_baseline.py`.


## Usage

```zsh
pip install git+https://github.com/stefaccou/TIPA
```
