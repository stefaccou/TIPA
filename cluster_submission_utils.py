from pathlib import Path
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput


def find_master(par="Master_thesis"):
    """
    Search for a parent directory named 'Master_thesis'.

    Returns:
        Path: The Path object corresponding to the 'Master_thesis' directory.

    Raises:
        FileNotFoundError: If no parent directory named 'Master_thesis' is found.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == par:
            return parent
    raise FileNotFoundError("Could not find a parent directory named 'Master_thesis'")


def update_submission_log(experiments_dir: Path, submission_text: str) -> int:
    """
    Update the submission log in the experiments directory and return the new run count.

    Parameters:
        experiments_dir (Path): The experiments directory where the log file is stored.
        submission_text (str): A string representing the submission details.

    Returns:
        int: The new run count after updating the log.
    """
    log_file = experiments_dir / "submission_log.txt"
    # Count previous submissions if log exists; else initialize count as 0.
    if log_file.exists():
        run_count = sum(1 for line in open(log_file) if line.startswith(submission_text))
    else:
        run_count = 0

    # Increment the count for this submission.
    run_count += 1

    # Append the submission info to the log file.
    with open(log_file, "a") as f:
        f.write(f"{submission_text} #{run_count}: Run {run_count:03d}\n")

    return run_count


class CustomTokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, hidden_states, labels=None):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # logits shape: (batch_size, sequence_length, num_labels)
            # labels shape: (batch_size, sequence_length)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)
