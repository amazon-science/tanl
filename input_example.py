# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int                  # start index in the sentence
    end: int                    # end index in the sentence
    type: Optional[EntityType] = None   # entity type
    id: Optional[int] = None    # id in the current training/test example

    def to_tuple(self):
        return self.type.natural, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType  # relation type
    head: Entity        # head of the relation
    tail: Entity        # tail of the relation

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class Intent:
    """
    The intent of an utterance.
    """
    short: str = None
    natural: str = None

    def __hash__(self):
        return hash(self.short)


@dataclass
class InputExample:
    """
    A single training/test example.
    """
    id: str                     # unique id in the dataset
    tokens: List[str]           # list of tokens (words)
    dataset: Optional[Dataset] = None   # dataset this example belongs to

    # entity-relation extraction
    entities: List[Entity] = None      # list of entities
    relations: List[Relation] = None   # list of relations
    intent: Optional[Intent] = None

    # event extraction
    triggers: List[Entity] = None               # list of event triggers

    # SRL
    sentence_level_entities: List[Entity] = None

    # coreference resolution
    document_id: str = None     # the id of the document this example belongs to
    chunk_id: int = None        # position in the list of chunks
    offset: int = None          # offset of this example in the document
    groups: List[List[Entity]] = None  # groups of entities

    # DST
    belief_state: Union[Dict[str, Any], str] = None
    utterance_tokens: str = None


@dataclass
class CorefDocument:
    """
    A document for the coreference resolution task.
    It has several input examples corresponding to chunks of the document.
    """
    id: str                     # unique id in the dataset
    tokens: List[str]           # list of tokens (words)
    chunks: List[InputExample]  # list of chunks for this document (the offset is an attribute of the InputExample)
    chunk_centers: List[int]    # list of the centers of the chunks (useful to find the chunk with largest context)
    groups: List[List[Entity]]  # coreference groups


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: Optional[List[int]] = None
