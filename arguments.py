# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )
    
    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    eval_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names. Defaults to the train datasets."}
    )

    train_split: str = field(
        default='train',
        metadata={"help": "The datasplit for training. Can be 'train', 'dev', 'test', etc."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    max_output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length (default is the same as input)"
        },
    )

    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_subset: float = field(
        default=1, metadata={"help": "The portion of training data to use"}
    )

    episodes: str = field(
        default='0', metadata={"help": "Episode indices -- a single number such as 3 or an interval such as 1-4\n"
                                       "The index is also used as random seeds and this setting is therefore used to "
                                       "repeat multiple experiments."}
    )

    num_beams: int = field(
        default=None,
        metadata={"help": "Number of beams for beam search during generation (only affects evaluation)"}
    )

    max_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "Maximum input sequence length at evaluation time (default is equal to max_seq_length)"
        },
    )

    max_output_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length at evaluation time (default is the same as input)"
        },
    )
    
    input_format: str = field(
        default=None, metadata={"help": "Input format"}
    )
    
    output_format: str = field(
        default=None, metadata={"help": "Output format"}
    )

    multitask: bool = field(
        default=False, metadata={"help": "If true, each input sentence is prepended with the dataset name"}
    )

    # few-shot arguments
    num_shots: int = field(
        default=None, metadata={"help": "number of shots (few-shot argument for the FewRel dataset)"}
    )

    num_ways: int = field(
        default=None, metadata={"help": "number of ways (few-shot argument for the FewRel dataset)"}
    )

    num_query: int = field(
        default=5, metadata={"help": "number of query examples (few-shot argument for the FewRel dataset)"}
    )

    # chunk arguments (used for the CoNLL2012 coreference resolution dataset)
    chunk_size: int = field(
        default=128, metadata={"help": "Size of document chunks"}
    )

    chunk_overlap: int = field(
        default=64, metadata={"help": "Size of overlap between consecutive chunks"}
    )

    chunk_size_eval: int = field(
        default=None, metadata={"help": "Size of document chunks during evaluation (default is equal to chunk_size)"}
    )

    chunk_overlap_eval: int = field(
        default=None, metadata={"help": "Size of overlap between consecutive chunks during evaluation "
                                        "(default is equal to chunk_overlap)"}
    )

    eval_nll: bool = field(
        default=False, metadata={"help": "Evaluate using NLL (only applicable to certain datasets)"}
    )
