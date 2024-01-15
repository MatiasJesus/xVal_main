import torch
from tqdm import tqdm
import pandas as pd
import numpy as np


def token_structure(entry, tokenizer):
    # Asegúrate de que 'entry' contenga 'input_ids' y 'numbers'
    input_ids = entry['input_ids']
    numbers = entry.get('numbers', None)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    token_info = []

    for i, token_id in enumerate(input_ids):
        token = tokens[i]
        text_value = tokenizer.decode([token_id])

        if token == '[NUM]' and numbers is not None:
            numeric_value = numbers[i]
            token_info.append(f"{token}({numeric_value})")
        else:
            token_info.append(text_value)

    return ' '.join(token_info)


def mask_numbers(sample, tokenizer, n_list):
    import copy

    mask_token = tokenizer.encode("[MASK]")[0]
    masked_sample = copy.deepcopy(sample)
    len_ = len(masked_sample["input_ids"])
    masked_sample["masked_numbers"] = copy.deepcopy(sample["numbers"])[:len_]
    masked_sample["numbers"] = masked_sample["numbers"][:len_]
    masked_sample["labels"] = sample["input_ids"]
    for n in n_list:
        masked_sample["input_ids"][n] = mask_token
        masked_sample["masked_numbers"][n] = 1.0
        # Next two lines are for calculating the correct mlm loss
        # tells the model to only look at the masked token for calculating x-entropy
        masked_sample["labels"] = list(0 * np.array(masked_sample["labels"]) - 100)
        masked_sample["labels"][n] = sample["input_ids"][n]
        masked_sample["ans"] = masked_sample["numbers"][n]
    masked_sample["text"] = tokenizer.decode(sample["input_ids"])
    masked_sample["masked_text"] = tokenizer.decode(masked_sample["input_ids"])
    return masked_sample


def mask_nth_number(sample, tokenizer, n):
    import copy

    mask_token = tokenizer.encode("[MASK]")[0]
    masked_sample = copy.deepcopy(sample)
    masked_sample["input_ids"][n] = mask_token
    len_ = len(masked_sample["input_ids"])
    masked_sample["masked_numbers"] = copy.deepcopy(sample["numbers"])[:len_]
    masked_sample["numbers"] = masked_sample["numbers"][:len_]
    masked_sample["labels"] = sample["input_ids"]
    masked_sample["masked_numbers"][n] = 1.0
    # Next two lines are for calculating the correct mlm loss
    # tells the model to only look at the masked token for calculating x-entropy
    masked_sample["labels"] = list(0 * np.array(masked_sample["labels"]) - 100)
    masked_sample["labels"][n] = sample["input_ids"][n]
    masked_sample["text"] = tokenizer.decode(sample["input_ids"])
    masked_sample["masked_text"] = tokenizer.decode(masked_sample["input_ids"])
    masked_sample["ans"] = masked_sample["numbers"][n]
    return masked_sample


### Each number
def predict(model, masked_sample, device="cuda"):
    model.eval()
    model.to(device)
    input = {
        "x": torch.tensor(masked_sample["input_ids"]).view(1, -1).to(device),
        # "y": torch.tensor(masked_sample["labels"]).view(1, -1).to(device),
        "x_num": torch.tensor(masked_sample["masked_numbers"]).view(1, -1).to(device),
        # "y_num": torch.tensor(masked_sample["masked_numbers"]).view(1, -1).to(device),
    }
    out = model(**input)
    return out


### Each row
def predict_numbers(model, sample, tokenizer, n_list, device, all_at_once=False):
    num_pred_list = []
    num_true_list = []
    if all_at_once:
        masked_sample = mask_numbers(sample, tokenizer, n_list)
        out = predict(model, masked_sample, device)
        for n in n_list:
            num_pred_list.append(out[1][0][n].item())
            num_true_list.append(masked_sample["numbers"][n])
    else:
        for n in n_list:
            masked_sample = mask_nth_number(sample, tokenizer, n)
            out = predict(model, masked_sample, device)
            num_pred_list.append(out[1][0][n].item())
            num_true_list.append(masked_sample["numbers"][n])

    return {
        "num_pred_list": num_pred_list,
        "num_true_list": num_true_list,
    }


### Run on whole dataset
def slow_eval_numbers(
    model, dataset, tokenizer, n_list, device, num_samples=None, all_at_once=False
):
    model.eval()
    model.to(device)

    if num_samples is None:
        num_samples = len(dataset)

    with torch.no_grad():
        out = []
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            out.append(
                predict_numbers(model, sample, tokenizer, n_list, device, all_at_once)
            )

    pd_out = pd.DataFrame(out)
    return pd_out
