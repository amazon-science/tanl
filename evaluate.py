# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Dict
import torch
import logging
import numpy as np
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from datasets import load_dataset


def get_avg_results(results: List[dict]) -> dict:
    """
    Compute average results and standard deviation from many episodes.
    """
    aggregate_results = {'num_episodes': len(results)}

    for key in results[0]:
        try:
            numbers = np.array([res[key] for res in results])
            aggregate_results[key] = (numbers.mean(), numbers.std())

        except:
            pass

    return aggregate_results


def print_results(results: dict):
    for key, value in results.items():
        s = f'{key.replace("_", " "):26} '

        if isinstance(value, (list, tuple)):
            mean, std = value
            s += f'{mean:.6f} ± {std:.6f}'
        elif isinstance(value, float):
            s += f'{value:.6f}'
        else:
            s += f'{value}'

        logging.info(s)


def evaluate(model, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, batch_size: int) -> Dict[str, float]:
    """
    Evaluate a model on some dataset.
    """
    model.eval()

    device = torch.device("cuda", gpu)
    model.to(device)

    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Num beams:  {data_args.num_beams}')
    logging.info(f'Max input length for evaluation:  {data_args.max_seq_length_eval}')
    logging.info(f'Max output length for evaluation: {data_args.max_output_seq_length_eval}')

    test_dataset = load_dataset(
        dataset_name, data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, split=split, seed=seed, shuffle=False, is_eval=True,
    )

    return test_dataset.evaluate_dataset(data_args=data_args, model=model, device=device, batch_size=batch_size)
