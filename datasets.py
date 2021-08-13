# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import bisect
import copy
import os
import logging
import json
from itertools import islice
from collections import Counter, defaultdict
import numpy as np
import random
import networkx as nx
from typing import Dict, List, Tuple, Set
import torch
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, CorefDocument
from base_dataset import BaseDataset
from utils import get_precision_recall_f1
from coreference_metrics import CorefAllMetrics
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS

DATASETS = {}


def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        train_subset: float = 1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False
):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        train_subset=train_subset,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
    )


class JointERDataset(BaseDataset):
    """
    Base class for datasets of joint entity and relation extraction.
    """
    entity_types = None
    relation_types = None

    natural_entity_types = None     # dictionary from entity types given in the dataset to the natural strings to use
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use

    default_output_format = 'joint_er'

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.entity_types, self.relation_types, self.examples, self.features = \
            d['entity_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)

    def load_schema(self):
        """
        Load entity and relation types.

        This is the default implementation which uses the dictionaries natural_entity_types and natural_relation_types.
        """
        if self.natural_entity_types is not None:
            self.entity_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_entity_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        This is the default implementation for datasets in the SpERT format
        (see https://github.com/markus-eberts/spert).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=entities[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )

                examples.append(example)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
        predicted_entities, predicted_relations = res[:2]
        if len(res) == 6:
            # the output format provides information about errors
            wrong_reconstruction, label_error, entity_error, format_error = res[2:]
        else:
            # in case the output format does not provide information about errors
            wrong_reconstruction = label_error = entity_error = format_error = False

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])

        # load ground truth entities
        gt_entities = set(entity.to_tuple() for entity in example.entities)
        gt_entities_no_type = set([entity[1:] for entity in gt_entities])

        # compute correct entities
        correct_entities = predicted_entities & gt_entities
        correct_entities_no_type = gt_entities_no_type & predicted_entities_no_type

        # load ground truth relations
        gt_relations = set(relation.to_tuple() for relation in example.relations)

        # compute correct relations
        correct_relations = predicted_relations & gt_relations

        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_entities_no_type) <= len(predicted_entities_no_type)
        assert len(correct_entities_no_type) <= len(gt_entities_no_type)

        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'entity_error': 1 if entity_error else 0,
            'format_error': 1 if format_error else 0,
            'gt_entities': len(gt_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'gt_entities_no_type': len(gt_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

        # add information about each entity/relation type so that we can compute the macro-F1 scores
        if self.entity_types is not None:
            for entity_type in self.entity_types.values():
                predicted = set(entity for entity in predicted_entities if entity[0] == entity_type.natural)
                gt = set(entity for entity in gt_entities if entity[0] == entity_type.natural)
                correct = predicted & gt
                res['predicted_entities', entity_type.natural] = len(predicted)
                res['gt_entities', entity_type.natural] = len(gt)
                res['correct_entities', entity_type.natural] = len(correct)

        if self.relation_types is not None:
            for relation_type in self.relation_types.values():
                predicted = set(relation for relation in predicted_relations if relation[0] == relation_type.natural)
                gt = set(relation for relation in gt_relations if relation[0] == relation_type.natural)
                correct = predicted & gt
                res['predicted_relations', relation_type.natural] = len(predicted)
                res['gt_relations', relation_type.natural] = len(gt)
                res['correct_relations', relation_type.natural] = len(correct)

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['gt_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['gt_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        if macro:
            # compute also entity macro scores
            for entity_type in self.entity_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_entities', entity_type.natural],
                    num_predicted=results['predicted_entities', entity_type.natural],
                    num_gt=results['gt_entities', entity_type.natural],
                )
                entity_precision_by_type.append(precision)
                entity_recall_by_type.append(recall)
                entity_f1_by_type.append(f1)

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'entity_error': results['entity_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
        }

        if macro:
            res.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return res


@register_dataset
class Conll04Dataset(JointERDataset):
    """
    CoNLL04 dataset (joint entity and relation extraction).

    Downloaded using https://github.com/markus-eberts/spert/blob/master/scripts/fetch_datasets.sh
    """
    name = 'conll04'

    natural_entity_types = {
        'Loc': 'location',
        'Org': 'organization',
        'Peop': 'person',
        'Other': 'other',
    }

    natural_relation_types = {
        'Work_For': 'works for',
        'Kill': 'kills',
        'OrgBased_In': 'organization based in',
        'Live_In': 'lives in',
        'Located_In': 'located in'
    }


@register_dataset
class ADEDataset(JointERDataset):
    """
    ADE dataset (joint entity and relation extraction).

    Downloaded using https://github.com/markus-eberts/spert/blob/master/scripts/fetch_datasets.sh
    """
    name = 'ade'

    natural_entity_types = {
        'Adverse-Effect': 'disease',
        'Drug': 'drug',
    }

    natural_relation_types = {
        'Adverse-Effect': 'effect',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        We decide which split to use based on the seed.
        In this way, running episodes 1-10 has the effect of running on all 10 different splits once.
        """
        if seed is None:
            i = 0
        else:
            i = seed % 10

        if split == 'train':
            return super().load_data_single_split(f'split_{i}_train', seed)

        elif split == 'dev':
            return []

        elif split == 'test':
            return super().load_data_single_split(f'split_{i}_test', seed)

    def evaluate_dataset(self, *args, **kwargs):
        """
        Evaluate model on this dataset.

        We include the macro entity scores, since it is standard to report them.
        """
        return super().evaluate_dataset(*args, **kwargs, macro=True)


@register_dataset
class NYTDataset(JointERDataset):
    """
    NYT dataset (joint entity and relation extraction).

    Downloaded from https://github.com/yubowen-ph/JointER/tree/master/dataset/NYT-multi/data
    """
    name = 'nyt'

    natural_entity_types = {
        'PERSON': 'person',
        'LOCATION': 'location',
        'ORGANIZATION': 'organization',
    }

    @staticmethod
    def to_natural_relation_type(relation_type: str) -> str:
        # for example, '/people/person/place_of_birth' -> place of birth
        return relation_type.split('/')[-1].replace('_', ' ')

    def load_schema(self):
        """
        Load entity and relation types.
        """
        # entity types are given explicitly in self.natural_entity_types
        super().load_schema()

        # load relation types from file
        with open(os.path.join(self.data_dir(), f'schemas.json'), 'r') as f:
            types = json.load(f)

            # self.entity_types = {name: EntityType(
            #     natural=self.to_natural_entity_type(name),
            # ) for name in types[2].values()}

            self.relation_types = {name: RelationType(
                natural=self.to_natural_relation_type(name)
            ) for name in types[0].values()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = []
                relations = []

                for y in x['spo_details']:
                    entity1_start, entity1_end, entity1_type, relation_type, \
                        entity2_start, entity2_end, entity2_type = y

                    entity1 = Entity(type=self.entity_types[entity1_type], start=entity1_start, end=entity1_end)
                    entity2 = Entity(type=self.entity_types[entity2_type], start=entity2_start, end=entity2_end)

                    try:
                        i1 = entities.index(entity1)
                    except ValueError:
                        # add entity to the list
                        i1 = len(entities)
                        entities.append(entity1)

                    try:
                        i2 = entities.index(entity2)
                    except ValueError:
                        # add entity to the list
                        i2 = len(entities)
                        entities.append(entity2)

                    relation = Relation(
                        type=self.relation_types[relation_type], head=entities[i1], tail=entities[i2],
                    )

                    relations.append(relation)

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )

                examples.append(example)

        return examples


@register_dataset
class ACE2005REDataset(JointERDataset):
    """
    ACE2005 dataset (joint entity and relation extraction.

    Processed using https://github.com/luanyi/DyGIE/tree/master/preprocessing
    """
    name = 'ace2005_joint_er'

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }

    natural_relation_types = {
        'PHYS': 'located in',
        'ART': 'artifact',
        'ORG-AFF': 'employer',
        'GEN-AFF': 'affiliation',
        'PER-SOC': 'social',
        'PART-WHOLE': 'part of',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        examples = []
        num_documents = 0
        num_entities = 0
        num_relations = 0

        with open(file_path, 'r') as f:
            for j, l in enumerate(f):
                document = json.loads(l)
                num_documents += 1
                offset = 0

                for i, tokens in enumerate(document['sentences']):
                    num_entities += len(document['ner'][i])
                    num_relations += len(document['relations'][i])

                    if len(document['ner'][i]) > 0:
                        entities = [
                            Entity(type=self.entity_types[entity_type], start=start-offset, end=end-offset+1)
                            for start, end, entity_type in document['ner'][i]
                        ]

                        relations = []

                        skip = False
                        for start1, end1, start2, end2, relation_type in document['relations'][i]:
                            # find entities
                            if len([e for e in entities if e.start == start1-offset and e.end == end1-offset+1]) > 1 \
                                    or \
                                    len([e for e in entities if e.start == start2-offset and e.end == end2-offset+1]) \
                                    > 1:
                                skip = True
                                break

                            [head] = [e for e in entities if e.start == start1-offset and e.end == end1-offset+1]
                            [tail] = [e for e in entities if e.start == start2-offset and e.end == end2-offset+1]

                            relations.append(
                                Relation(type=self.relation_types[relation_type], head=head, tail=tail)
                            )

                        if not skip:
                            example = InputExample(
                                id=f'{split}-{j}-{i}',
                                tokens=tokens,
                                entities=entities,
                                relations=relations,
                            )
                            examples.append(example)

                    offset += len(tokens)

        logging.info(f'Constructed {len(examples)} examples (from {num_documents} documents) for {self.name} ({split})')
        return examples


class NERDataset(JointERDataset):
    """
    Base class for NER datasets.
    """

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.txt')

        raw_examples = []
        tokens = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        raw_examples.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    splits = line.split()
                    tokens.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].strip()
                        if label == 'O':
                            label = None
                        labels.append(label)
                    else:
                        labels.append(None)

            if tokens:
                raw_examples.append((tokens, labels))

        logging.info(f"Loaded {len(raw_examples)} sentences for split {split} of {self.name}")

        examples = []
        for i, (tokens, labels) in enumerate(raw_examples):
            assert len(tokens) == len(labels)

            # process labels
            entities = []

            current_entity_start = None
            current_entity_type = None

            for j, label in enumerate(labels + [None]):
                previous_label = labels[j-1] if j > 0 else None
                if (label is None and previous_label is not None) \
                        or (label is not None and previous_label is None) \
                        or (label is not None and previous_label is not None and (
                            label[2:] != previous_label[2:] or label.startswith('B-') or label.startswith('S-')
                        )):
                    if current_entity_start is not None:
                        # close current entity
                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[current_entity_type],
                            start=current_entity_start,
                            end=j,
                        ))

                        current_entity_start = None
                        current_entity_type = None

                    if label is not None:
                        # a new entity begins
                        current_entity_start = j
                        assert any(label.startswith(f'{prefix}-') for prefix in 'BIS')
                        current_entity_type = label[2:]
                        assert current_entity_type in self.entity_types

            example = InputExample(
                id=f'{split}-{i}',
                tokens=tokens,
                entities=entities,
                relations=[],
            )

            examples.append(example)

        return examples

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, and return entity metrics only.
        """
        results = super().evaluate_dataset(data_args, model, device, batch_size, macro=macro)
        return {k: v for k, v in results.items() if k.startswith('entity') and k != 'entity_error'}


@register_dataset
class CoNLL03Dataset(NERDataset):
    """
    CoNLL03 dataset (NER).
    """
    name = 'conll03'

    natural_entity_types = {
        'LOC': 'location',
        'MISC': 'miscellaneous',
        'ORG': 'organization',
        'PER': 'person',
    }


@register_dataset
class OntonotesDataset(NERDataset):
    """
    Ontonotes dataset (NER).
    """
    name = 'ontonotes'

    natural_entity_types = {
        'CARDINAL': 'cardinal',
        'DATE': 'date',
        'EVENT': 'event',
        'FAC': 'facility',
        'GPE': 'country city state',
        'LANGUAGE': 'language',
        'LAW': 'law',
        'LOC': 'location',
        'MONEY': 'monetary',
        'NORP': 'nationality religious political group',
        'ORDINAL': 'ordinal',
        'ORG': 'organization',
        'PERCENT': 'percent',
        'PERSON': 'person',
        'PRODUCT': 'product',
        'QUANTITY': 'quantity',
        'TIME': 'time',
        'WORK_OF_ART': 'work_of_art',
    }


@register_dataset
class SnipsDataset(NERDataset):
    name = 'snips'

    default_output_format = 'joint_icsl'

    natural_entity_types = {
        'entity_name': 'entity name',
        'playlist_owner': 'playlist owner',
        'playlist': 'playlist',
        'music_item': 'music item',
        'artist': 'artist',
        'party_size_description': 'party size description',
        'party_size_number': 'party size number',
        'restaurant_type': 'restaurant type',
        'spatial_relation': 'spatial relation',
        'state': 'state',
        'cuisine': 'cuisine',
        'poi': 'poi',
        'country': 'country',
        'city': 'city',
        'timeRange': 'time range',
        'facility': 'facility',
        'served_dish': 'served dish',
        'condition_description': 'condition description',
        'geographic_poi': 'geographic poi',
        'condition_temperature': 'condition temperature',
        'current_location': 'current location',
        'album': 'album',
        'service': 'service',
        'sort': 'sort',
        'track': 'track',
        'year': 'year',
        'object_name': 'object name',
        'rating_value': 'rating value',
        'best_rating': 'best rating',
        'rating_unit': 'rating unit',
        'object_select': 'object select',
        'object_part_of_series_type': 'object part of series type',
        'movie_name': 'movie name',
        'location_name': 'location name',
        'object_location_type': 'object location type',
        'movie_type': 'movie type'
    }

    natural_intent_types = {
        'AddToPlaylist': 'add to playlist',
        'BookRestaurant': 'book restaurant',
        'GetWeather': 'get weather',
        'PlayMusic': 'play music',
        'RateBook': 'rate book',
        'SearchCreativeWork': 'search creative work',
        'SearchScreeningEvent': 'search screening event'
    }

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                current_entity = Entity(
                    type=EntityType(
                        short=tag_type,
                        natural=self.natural_entity_types[tag_type]
                        if tag_type in self.natural_entity_types else tag_type
                    ),
                    start=ii,
                    end=ii+1,
                )
                entities.append(current_entity)
            elif el.startswith('I-'):
                current_entity.end = ii + 1
        return entities


    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.tsv')
        examples = []

        with open(file_path, 'r') as f:
            data = f.readlines()[1:]    # skip header
            for id, example in enumerate(data):
                uid, cid, turn, author, utterance, short_intent, act, slot_labels = example.strip().split('\t')
                tokens = utterance.split()

                slot_labels = slot_labels.split()
                entities = self.convert_bio_to_entities(slot_labels)

                intent = Intent(
                    short=short_intent,
                    natural=self.natural_intent_types[short_intent]
                    if short_intent in self.natural_intent_types.keys() else short_intent
                )

                example = InputExample(
                    id=f'{split}-{id}',
                    tokens=tokens,
                    intent=intent,
                    entities=entities,
                )

                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }
        self.intents = {
            example.intent.natural: example.intent
            for example in examples
        }

        return examples


    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
        )
        predicted_intent, predicted_entities, wrong_reconstruction, label_error, format_error = res

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])

        # load ground truth entities
        gt_entities = set(entity.to_tuple() for entity in example.entities)
        gt_entities_no_type = set([entity[1:] for entity in gt_entities])

        # compute correct entities
        correct_entities = predicted_entities & gt_entities
        correct_entities_no_type = gt_entities_no_type & predicted_entities_no_type

        # load ground truth intent
        gt_intent = example.intent

        # print(f"Ground truth: {gt_intent} ||| Predicted: {predicted_intent}")

        # compute correct intent
        correct_intent = int(predicted_intent == gt_intent.natural)


        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_entities_no_type) <= len(predicted_entities_no_type)
        assert len(correct_entities_no_type) <= len(gt_entities_no_type)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'format_error': 1 if format_error else 0, 
            'predicted_intent': 1 if len(predicted_intent) > 0 else 0,
            'gt_intent': 1,
            'correct_intent': correct_intent,
            'gt_entities': len(gt_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'gt_entities_no_type': len(gt_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
        })
        
        if self.intents is not None:
            for intent_type in self.intents.values():
                predicted = int(predicted_intent == intent_type.natural)
                gt = int(gt_intent.natural == intent_type.natural)
                correct = int(predicted_intent == gt_intent.natural)
                res['predicted_intent', intent_type.natural] = predicted
                res['gt_intent', intent_type.natural] = gt
                res['correct_intent', intent_type.natural] = correct

        # add information about each entity/relation type so that we can compute the macro-F1 scores
        if self.entity_types is not None:
            for entity_type in self.entity_types.values():
                predicted = set(entity for entity in predicted_entities if entity[0] == entity_type.natural)
                gt = set(entity for entity in gt_entities if entity[0] == entity_type.natural)
                correct = predicted & gt
                res['predicted_entities', entity_type.natural] = len(predicted)
                res['gt_entities', entity_type.natural] = len(gt)
                res['correct_entities', entity_type.natural] = len(correct)

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['gt_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['gt_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        if macro:
            # compute also entity macro scores
            for entity_type in self.entity_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_entities', entity_type.natural],
                    num_predicted=results['predicted_entities', entity_type.natural],
                    num_gt=results['gt_entities', entity_type.natural],
                )
                entity_precision_by_type.append(precision)
                entity_recall_by_type.append(recall)
                entity_f1_by_type.append(f1)

        intent_precision, intent_recall, intent_f1 = get_precision_recall_f1(
            num_correct=results['correct_intent'],
            num_predicted=results['predicted_intent'],
            num_gt=results['gt_intent']
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'intent_precision': intent_precision,
            'intent_recall': intent_recall,
            'intent_f1': intent_f1,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
        }

        if macro:
            res.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return res


@register_dataset
class ATISDataset(SnipsDataset):
    name = 'atis'

    default_output_format = 'joint_icsl'

    natural_entity_types = {
        'fromloc': 'from', 'toloc': 'to',
        'city_name': 'city', 'state_code': 'state code', 'state_name': 'state name',
        'country_name': 'country name',
        'airport_code': 'airport code', 'airport_name': 'airport name',
        'depart_date': 'depart date', 'arrive_date': 'arrive date',
        'depart_time': 'depart time', 'arrive_time': 'arrive time',
        'return_date': 'return date', 'return_time': 'return time',
        'day_number': 'day number', 'day_name': 'day name', 'days_code': 'days code',
        'month_name': 'month', 'year': 'year',
        'date_relative': 'relative date', 'today_relative': 'relative today',
        'period_of_day': 'period of day', 'time_relative': 'relative time',
        'time': 'time', 'start_time': 'start time', 'end_time': 'end time',
        'cost_relative': 'relative cost',
        'airline_name': 'airline name', 'airline_code': 'airline code',
        'class_type': 'class type',
        'round_trip': 'round trip',
        'fare_basis_code': 'fare basis code',
        'fare_amount': 'fare amount',
        'meal': 'meal', 'meal_code': 'meal code', 'meal_description': 'meal description',
        'flight_mod': 'flight modify', 'mod': 'modify', 'period_mod': 'period modify',
        'stoploc': 'stop location',
        'connect': 'connect',
        'flight_number': 'flight number', 'flight_time': 'flight time', 'flight_stop': 'flight stop',
        'flight_days': 'flight days',
        'aircraft_code': 'aircraft code',
        'or': 'or',
        'restriction_code': 'restriction code',
        'transport_type': 'transport type',
        'economy': 'economy'
    }

    natural_intent_types = {
        'atis_flight': 'flight',
        'atis_airfare': 'airfare',
        'atis_airline': 'airline',
        'atis_aircraft': 'aircraft',
        'atis_flight<DIV>atis_airfare': 'flight and airfare',
        'atis_abbreviation': 'abbreviation',
        'atis_ground_service': 'ground service',
        'atis_meal': 'meal',
        'atis_restriction': 'restriction',
        'atis_quantity': 'quantity',
        'atis_aircraft<DIV>atis_flight<DIV>atis_flight_no': 'aircraft and flight and flight number',
        'atis_airport': 'airport',
        'atis_ground_fare': 'ground fare',
        'atis_airline<DIV>atis_flight_no': 'airline and flight number',
        'atis_flight_time': 'flight time',
        'atis_flight_no': 'flight number',
        'atis_distance': 'distance',
        'atis_city': 'city',
        'atis_capacity': 'capacity',
        'atis_cheapest': 'cheapest',
        'atis_ground_service<DIV>atis_ground_fare': 'ground service and ground fare'
    }

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                if '.' in tag_type:
                    natural = ' '.join([self.natural_entity_types[tag_part]
                            if tag_part in self.natural_entity_types else tag_part
                            for tag_part in tag_type.split('.')])
                else:
                    natural = self.natural_entity_types[tag_type] if tag_type in self.natural_entity_types else tag_type
                current_entity = Entity(
                    type=EntityType(
                        short=tag_type,
                        natural=natural
                    ),
                    start=ii,
                    end=ii+1,
                )
                entities.append(current_entity)
            elif el.startswith('I-'):
                current_entity.end = ii + 1
        return entities


@register_dataset
class ACE2005NERDataset(NERDataset):
    """
    ACE2005 dataset (NER).

    Downloaded from https://github.com/ShannonAI/mrc-for-flat-nested-ner/
    """
    name = 'ace2005_ner'

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.ner.json')
        examples = []

        with open(file_path, 'r') as f:
            data = json.load(f)

            for i, x in enumerate(data):
                tokens = x['context'].split()
                entities = []

                for entity_type, l in x['label'].items():
                    for start_end in l:
                        start, end = map(int, start_end.split(';'))
                        end += 1

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[entity_type],
                            start=start,
                            end=end,
                        ))

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=[],
                )

                examples.append(example)

        logging.info(f"Loaded {len(examples)} sentences for split {split} of {self.name}")
        return examples


@register_dataset
class GENIADataset(NERDataset):
    """
    GENIA dataset (NER).
    """
    name = 'genia'

    natural_entity_types = {
        'G#DNA': 'DNA',
        'G#RNA': 'RNA',
        'G#cell_line': 'cell line',
        'G#cell_type': 'cell type',
        'G#protein': 'protein',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.data')
        examples = []

        with open(file_path, 'r') as f:
            for i, (raw_sentence, _, raw_entities, _) in enumerate(iter(lambda: tuple(islice(f, 4)), ())):
                tokens = raw_sentence.strip().split()
                entities = []

                for raw_entity in raw_entities.strip().split('|'):
                    if len(raw_entity) > 0:
                        span, entity_type = raw_entity.split()
                        start, end = map(int, span.split(','))

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[entity_type],
                            start=start,
                            end=end,
                        ))

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=[],
                )

                examples.append(example)

        logging.info(f"Loaded {len(examples)} sentences for split {split} of {self.name}")
        return examples


@register_dataset
class ACE2005EventTriggerDataset(JointERDataset):
    """
    ACE 2005 dataset (event extraction), trigger extraction component.
    """
    name = 'ace2005event_trigger'
    data_name = 'ace2005event'

    relation_schemas = None

    def load_schema(self):
        types_file_name = os.path.join(self.data_dir(), f'{self.data_name}_types.json')
        with open(types_file_name, 'r') as f:
            types = json.load(f)

            self.entity_types = {name: EntityType(
                short=name,
                natural=x['verbose'],
            ) for name, x in types['entities'].items()}

            self.relation_types = {name: RelationType(
                short=name,
                natural=x['verbose'],
            ) for name, x in types['relations'].items()}

        schema_file_name = os.path.join(self.data_dir(), f'{self.data_name}_schema.json')
        with open(schema_file_name, 'r') as f:
            schema = json.load(f)
            self.relation_schemas = dict()
            for trigger_type, role_types in schema.items():
                trigger_type = self.entity_types[trigger_type].natural
                self.relation_schemas[trigger_type] = \
                    set(self.relation_types[role_type].natural for role_type in role_types)

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        file_path = os.path.join(self.data_dir(), f'{self.data_name}_{split}.json')
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                triggers = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['triggers'])
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=triggers,
                    relations=[],
                )

                examples.append(example)

        return examples


@register_dataset
class ACE2005EventArgumentDataset(ACE2005EventTriggerDataset):
    """
    ACE 2005 dataset (event extraction), argument extraction component.
    """
    name = 'ace2005event_argument'
    data_name = 'ace2005event'

    default_input_format = 'ace2005_event_with_trigger'
    default_output_format = 'ace2005_event'

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                num_triggers = x['triggers']
                for trigger_id in range(min(1, len(num_triggers))):
                    entities = [
                        Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                        for j, y in enumerate(x['entities'])
                    ]

                    triggers = [
                        Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                        for j, y in enumerate(x['triggers'][trigger_id:trigger_id+1]) if len(x['triggers']) > 0
                    ]
                    assert len(triggers) <= 1, 'no more than 1 trigger'

                    relations = [
                        # here we take the trigger as the tail entity of the relation
                        Relation(
                            type=self.relation_types[y['type']], head=entities[y['head']], tail=triggers[y['tail']]
                        )
                        for y in x['relations'] if y['tail'] == trigger_id and len(num_triggers) > 0
                    ]
                    tokens = x['tokens']

                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        entities=entities,
                        triggers=triggers,
                        relations=relations,
                    )

                    examples.append(example)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract natural name of entity and relation types
        predicted_entities, predicted_relations, wrong_reconstruction = \
            self.output_format.run_inference(
                example,
                output_sentence,
                entity_types=self.entity_types,
                relation_types=self.relation_types,
            )

        # filter relation tuples for argument classification
        # since we don't need the entity type to be predicted correct

        def filter_relation_tuple(relation_tuple):
            return relation_tuple[0], relation_tuple[1][1:], relation_tuple[2]

        gt_relations = set(filter_relation_tuple(relation.to_tuple()) for relation in example.relations)
        gt_relations_no_type = set([relation[1:] for relation in gt_relations])

        # load ground truth relations that only have valid relations (exist in relation_schema)
        filtered_predicted_relations = set()
        for relation in predicted_relations:
            if relation[2][0] in self.relation_schemas and relation[0] in self.relation_schemas[relation[2][0]]:
                filtered_predicted_relations.add(filter_relation_tuple(relation))

        predicted_relations = filtered_predicted_relations
        predicted_relations_no_type = set(relation[1:] for relation in predicted_relations)

        # compute correct relations
        correct_relations = predicted_relations & gt_relations
        correct_relations_no_type = predicted_relations_no_type & gt_relations_no_type

        return Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
            'gt_relations_no_type': len(gt_relations_no_type),
            'predicted_relations_no_type': len(predicted_relations_no_type),
            'correct_relations_no_type': len(correct_relations_no_type),
        })

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        relation_precision_no_type, relation_recall_no_type, relation_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_relations_no_type'],
            num_predicted=results['predicted_relations_no_type'],
            num_gt=results['gt_relations_no_type'],
        )

        res = {
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'relation_precision_no_type': relation_precision_no_type,
            'relation_recall_no_type': relation_recall_no_type,
            'relation_f1_no_type': relation_f1_no_type,
            'num_gt_triggers': results['gt_entities'],
            'num_pred_triggers': results['predicted_entities'],
            'num_gt_relations': results['gt_relations'],
            'num_pred_relations': results['predicted_relations'],
        }

        return res


@register_dataset
class ACE2005EventDataset(ACE2005EventArgumentDataset):
    """
    ACE 2005 dataset (event extraction), for evaluation only.
    """
    name = 'ace2005event'
    task_descriptor = 'ace2005event_trigger'
    default_input_format = 'plain'
    default_output_format = 'joint_er'
    argument_input_format = 'ace2005_event_with_trigger'
    argument_output_format = 'ace2005_event'

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")
            
            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]
                
                triggers = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['triggers'])
                ]

                relations = [
                    # the trigger is the tail, and the entity is the head
                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=triggers[y['tail']]
                    )
                    for y in x['relations']
                ]
                
                tokens = x['tokens']
                
                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    triggers=triggers,
                    relations=relations,
                )
                
                examples.append(example)
        
        return examples
    
    def evaluate_argument(self, output_format, example_argument_single_trigger: InputExample, example: InputExample,
                          argument_output_sentence: str) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """
        Perform argument prediction.
        """
        predicted_entities, predicted_relations, wrong_reconstruction = \
            output_format.run_inference(example_argument_single_trigger,
                                        argument_output_sentence,
                                        entity_types=self.entity_types,
                                        relation_types=self.relation_types)
        
        # filter relation tuples for argument classification
        # since we don't need the entity type to be predicted correct

        def filter_relation_tuple(relation_tuple):
            return relation_tuple[0], relation_tuple[1][1:], relation_tuple[2]

        gt_relations = set(filter_relation_tuple(relation.to_tuple()) for relation in example.relations)

        # load ground truth relations to only have relations that are valid (exist in relation_schema)
        filtered_predicted_relations = set()
        for relation in predicted_relations:
            if relation[2][0] in self.relation_schemas and relation[0] in self.relation_schemas[relation[2][0]]:
                filtered_predicted_relations.add(filter_relation_tuple(relation))
        
        predicted_relations = filtered_predicted_relations
        
        # compute correct relations
        correct_relations = predicted_relations & gt_relations
        
        return predicted_relations, gt_relations, correct_relations

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, trigger_output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            # phase 1: trigger prediction
            trigger_output_format = self.output_format
            predicted_triggers = \
                trigger_output_format.run_inference(
                    example,
                    trigger_output_sentence,
                    entity_types=self.entity_types,
                    relation_types=self.relation_types,
                )[0]
            gt_triggers = set(trigger.to_tuple() for trigger in example.triggers)
            correct_triggers = predicted_triggers & gt_triggers
            predicted_triggers_notype = set()
            gt_triggers_notype = set()
            # trigger tuple format: (type, start, end) -- resetting all types to the same as 'TYPE'
            for trig in predicted_triggers:
                trig_list = list(trig)
                trig_list[0] = 'TYPE'
                predicted_triggers_notype.add(tuple(trig_list))
            for trig in gt_triggers:
                trig_list = list(trig)
                trig_list[0] = 'TYPE'
                gt_triggers_notype.add(tuple(trig_list))
            correct_triggers_notype = predicted_triggers_notype & gt_triggers_notype
            
            # phase 2: argument classification
            all_gt_relations, all_predicted_relations, all_correct_relations = set(), set(), set()
            for trigger in predicted_triggers:
                example_argument_single_trigger = copy.deepcopy(example)
                trigger_type = None
                for trigger_type in self.entity_types:
                    if self.entity_types[trigger_type].natural == trigger[0]: break
                example_argument_single_trigger.triggers = [
                    Entity(type=self.entity_types[trigger_type], start=trigger[1], end=trigger[2])]

                argument_input_format = INPUT_FORMATS[self.argument_input_format]()
                argument_output_format = OUTPUT_FORMATS[self.argument_output_format]()
                example_input = argument_input_format.format_input(example_argument_single_trigger, multitask=True,
                                                                   task_descriptor=ACE2005EventArgumentDataset.name)
                example_input_ids = self.tokenizer.batch_encode_plus(
                    [example_input],
                    max_length=data_args.max_seq_length,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True
                )
                argument_output = model.generate(
                    example_input_ids['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams,
                )[0]  # only one sample
                argument_output_sentence = self.tokenizer.decode(argument_output, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)

                gt_relations, predicted_relations, correct_relations = \
                    self.evaluate_argument(argument_output_format, example_argument_single_trigger, example,
                                           argument_output_sentence)
                all_gt_relations = all_gt_relations.union(gt_relations)
                all_predicted_relations = all_predicted_relations.union(predicted_relations)
                all_correct_relations = all_correct_relations.union(correct_relations)
                
            all_predicted_relations_notype = set()
            all_gt_relations_notype = set()
            for rel in all_predicted_relations:
                rel_list = list(rel)
                rel_list[0] = 'TYPE'
                all_predicted_relations_notype.add(tuple(rel_list))
            for rel in all_gt_relations:
                rel_list = list(rel)
                rel_list[0] = 'TYPE'
                all_gt_relations_notype.add(tuple(rel_list))

            all_correct_relations_notype = all_predicted_relations_notype & all_gt_relations_notype
            res = Counter({
                'num_sentences': 1,
                'gt_triggers': len(gt_triggers),
                'predicted_triggers': len(predicted_triggers),
                'correct_triggers': len(correct_triggers),
                'correct_triggers_notype': len(correct_triggers_notype),
                'predicted_relations': len(all_predicted_relations),
                'gt_relations': len(all_gt_relations),
                'correct_relations': len(all_correct_relations),
                'correct_relations_notype': len(all_correct_relations_notype)
            })
            
            results += res
        
        trigger_precision, trigger_recall, trigger_f1 = get_precision_recall_f1(
            num_correct=results['correct_triggers'],
            num_predicted=results['predicted_triggers'],
            num_gt=results['gt_triggers'],
        )
        trigger_precision_notype, trigger_recall_notype, trigger_f1_notype = get_precision_recall_f1(
            num_correct=results['correct_triggers_notype'],
            num_predicted=results['predicted_triggers'],
            num_gt=results['gt_triggers'],
        )
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )
        relation_precision_notype, relation_recall_notype, relation_f1_notype = get_precision_recall_f1(
            num_correct=results['correct_relations_notype'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )
        
        full_results = {
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'relation_precision_notype': relation_precision_notype,
            'relation_recall_notype': relation_recall_notype,
            'relation_f1_notype': relation_f1_notype,
            'trigger_precision': trigger_precision,
            'trigger_recall': trigger_recall,
            'trigger_f1': trigger_f1,
            'trigger_precision_notype': trigger_precision_notype,
            'trigger_recall_notype': trigger_recall_notype,
            'trigger_f1_notype': trigger_f1_notype,
        }
        
        return full_results


@register_dataset
class CoNLL12CorefDataset(BaseDataset):
    """
    CoNLL2012 dataset (coreference resolution).
    """
    name = 'conll12_coref'
    default_output_format = 'coref'

    documents = None    # list of documents

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        self.documents = {}
        examples = []

        if self.is_eval:
            chunk_size = self.data_args.chunk_size_eval
            chunk_overlap = self.data_args.chunk_overlap_eval
        else:
            chunk_size = self.data_args.chunk_size
            chunk_overlap = self.data_args.chunk_overlap

        with open(file_path, 'r') as f:
            for i, l in enumerate(f):
                raw_document = json.loads(l)
                document_id = f'{split}-{i}'

                tokens_data = raw_document['preprocessing']['segments']['tokens']
                tokens = [x['extent'] for x in tokens_data]
                tokens_start_char = [x['start'] for x in tokens_data]
                tokens_end_char = [x['end'] for x in tokens_data]

                groups = []
                for raw_group in raw_document['annotations']['coreference']['groups']:
                    mentions = []
                    for raw_mention in raw_group['mentions']:
                        # find start and end tokens
                        start = bisect.bisect_left(tokens_start_char, raw_mention['start'])
                        end = bisect.bisect_left(tokens_end_char, raw_mention['end']) + 1
                        mentions.append(Entity(start=start, end=end))

                    groups.append(mentions)

                # create chunks
                chunks = []
                pos = 0
                chunk_id = 0
                while pos < len(tokens):
                    # create a chunk starting at this position
                    chunk_tokens = tokens[pos:pos+chunk_size]

                    chunk_groups = []
                    for group in groups:
                        mentions = [
                            Entity(start=mention.start-pos, end=mention.end-pos, type=mention.type)
                            for mention in group
                            if mention.start >= pos and mention.end <= pos + chunk_size
                        ]
                        if len(mentions) >= 2:
                            chunk_groups.append(mentions)

                    example = InputExample(
                        id=f'{split}-{i}-{chunk_id}',
                        tokens=chunk_tokens,
                        offset=pos,
                        groups=chunk_groups,
                        document_id=document_id,
                        chunk_id=chunk_id,
                    )

                    examples.append(example)
                    chunks.append(example)

                    if pos + chunk_size >= len(tokens):
                        # this chunk arrives until the end, so we can stop
                        break

                    pos += chunk_size - chunk_overlap
                    chunk_id += 1

                self.documents[document_id] = CorefDocument(
                    id=document_id,
                    tokens=tokens,
                    groups=groups,
                    chunks=chunks,
                    chunk_centers=[example.offset + len(example.tokens) // 2 for example in chunks]
                )

        logging.info(f"Loaded {len(self.documents)} documents split in {len(examples)} chunks"
                     f" for split {split} of {self.name}")

        return examples

    @staticmethod
    def get_document_predictions(chunk_data: List[List[tuple]]) -> List[List[Tuple[int, int]]]:
        """
        Aggregate predictions for each chunk into document-level predictions.
        """
        all_edges = set(x for l in chunk_data for x in l)

        graph = nx.Graph()
        graph.add_edges_from(all_edges)

        processed_groups = []
        for component in nx.connected_components(graph):
            processed_group = []
            for start, end in sorted(component, key=lambda x: (x[0], -x[1])):
                # add this entity if it does not overlap with the previous one
                if len(processed_group) == 0 or start >= processed_group[-1][1]:
                    processed_group.append((start, end))

            processed_groups.append(processed_group)

        return [[(start, end) for start, end in group] for group in processed_groups]

    def evaluate_dataset(self, data_args, model, device, batch_size=8, macro=False, by_relation_type=False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        documents_to_chunk_data = defaultdict(list)
        predictions = {}

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            document_id = example.document_id

            data = self.output_format.run_inference(
                example=example,
                output_sentence=output_sentence,
            )

            # add offset to all indices
            offset = example.offset
            data = [tuple(tuple(y + offset for y in x) for x in z) for z in data if z[1] is not None]

            documents_to_chunk_data[document_id].append(data)

            if len(documents_to_chunk_data[document_id]) == len(self.documents[document_id].chunks):
                # process predictions for this document
                predictions[document_id] = self.get_document_predictions(documents_to_chunk_data[document_id])

        predictions_list = []
        labels_list = []
        for document_id, document in self.documents.items():
            predictions_list.append(predictions[document_id])
            labels_list.append([
                [(entity.start, entity.end) for entity in group]
                for group in document.groups
            ])

        metrics = CorefAllMetrics().get_all_metrics(labels_list, predictions_list)
        return {
            f'{metric_name}_{x}': v
            for metric_name, metric_values in metrics['micro'].items()
            for x, v in metric_values.items()
        }


class RelationClassificationDataset(JointERDataset):
    """
    Base class for relation classification datasets, implementing NLL inference.
    """

    def nll_inference(self, example: InputExample, relation_types: List[RelationType], model=None,
                      tokenizer=None) -> RelationType:
        """
        Run inference on a single example of this dataset, searching for the relation which maximizes the likelihood
        of the corresponding output sentence.
        """
        formatted_input = [self.input_format.format_input(example)]
        scores = []

        x = tokenizer.batch_encode_plus(
            formatted_input,
            max_length=self.max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        for relation_type in relation_types:
            # make a copy of the given example, substituting the correct relation with the current one
            candidate_example = copy.deepcopy(example)
            candidate_example.relations[0].type = relation_type

            # construct augmented natural language output
            formatted_output = [self.output_format.format_output(candidate_example)]
            y = tokenizer.batch_encode_plus(
                formatted_output,
                max_length=self.max_output_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
            )

            # run through the model to compute the likelihood
            res = model(
                input_ids=x['input_ids'].to(model.device),
                attention_mask=x['attention_mask'].to(model.device),
                labels=y['input_ids'].to(model.device)
            )
            scores.append(res[0].cpu().detach())    # res[0] is the log likelihood

        scores = np.array(scores)
        min_idx = scores.argmin()

        return relation_types[min_idx]


@register_dataset
class FewRelFull(RelationClassificationDataset):
    """
    Full FewRel dataset (relation classification), not episodic.

    Data was downloaded from https://github.com/thunlp/FewRel/tree/master/data
    """
    name = 'FewRelFull'
    data_name = 'FewRel'

    natural_entity_types = {
        'head': 'head',
        'tail': 'tail',
    }   # fake entity types corresponding to head and tail of a relation

    default_input_format = 'rel_input'
    default_output_format = 'rel_output'

    def load_schema(self):
        """
        Load relation types from the pid2name.json file provided with the dataset.
        """
        super().load_schema()   # this is to initialize the fake entity types 'head' and 'tail'

        with open(os.path.join(self.data_dir(), 'pid2name.json'), 'r') as f:
            data = json.load(f)
            self.relation_types = {
                short: RelationType(short=short, natural=description[0])
                for short, description in data.items()
            }

    def load_data_by_relation_type(self, split: str) -> Dict[str, List[InputExample]]:
        """
        Load data for a single split (train or dev) by relation type.

        This is useful for episodic training/evaluation, where we sample N classes at each episode.
        """
        examples_by_type = {}
        file_path = os.path.join(self.data_dir(), f'{self.data_name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            i = 0

            for type_id in data:
                assert type_id in self.relation_types
                relation_type = self.relation_types[type_id]
                examples = []

                for idx, _data in enumerate(data[type_id]):
                    tokens = _data['tokens']
                    head_entity = _data['h'][2][0]
                    tail_entity = _data['t'][2][0]

                    if len(head_entity) == 1:
                        head_entity = [head_entity[0], head_entity[0]]

                    if len(tail_entity) == 1:
                        tail_entity = [tail_entity[0], tail_entity[0]]

                    head_entity = Entity(id=None, type=self.entity_types['head'],
                                         start=head_entity[0], end=head_entity[1] + 1)

                    tail_entity = Entity(id=None, type=self.entity_types['tail'],
                                         start=tail_entity[0], end=tail_entity[1] + 1)

                    entities = [head_entity, tail_entity]

                    relations = [
                        Relation(
                            type=relation_type, head=head_entity, tail=tail_entity
                        )
                    ]

                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        entities=entities,
                        relations=relations,
                    )
                    examples.append(example)
                    i += 1

                examples_by_type[type_id] = examples

        return examples_by_type

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples_by_type = self.load_data_by_relation_type(split=split)
        examples = [example for x in examples_by_type.values() for example in x]
        return examples

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, and return relation metrics only.
        """
        results = super().evaluate_dataset(data_args, model, device, batch_size, macro=macro)
        return {k: v for k, v in results.items() if k.startswith('relation')}


@register_dataset
class FewRelEpisodic(FewRelFull):
    """
    Full FewRel dataset (relation classification), episodic.

    Episodic fine-tuning should happen after meta-training on the FewRelFull dataset.
    """
    name = 'FewRelEpisodic'

    default_input_format = 'rel_input'
    default_output_format = 'rel_output'

    target_relation_types = None    # the relation types involved in this episode (there is self.num_ways of them)

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples_by_type = self.load_data_by_relation_type(split='dev')   # we use the dev set for episodic experiments

        # set few-shot parameters
        num_ways, num_shots, num_queries = self.data_args.num_ways, self.data_args.num_shots, self.data_args.num_query

        # sample num_ways relation types for this particular episode
        self.target_relation_types = {
            type_id: self.relation_types[type_id]
            for type_id in random.sample(list(examples_by_type.keys()), num_ways)
        }

        logging.info(f'Target relation types for this few-shot episode: '
                     f'{[relation.natural for relation in self.target_relation_types.values()]}')

        support = []
        query = []

        # sample num_shots + num_queries examples from each relation type
        for type_id in self.target_relation_types:
            sampled_examples = random.sample(examples_by_type[type_id], num_shots + num_queries)
            support += sampled_examples[:num_shots]
            query += sampled_examples[num_shots:]

        if split == 'train':
            return support
        else:
            return query

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate a single example of this dataset, using NLL inference.
        """
        predicted_relation = self.nll_inference(
            example=example,
            relation_types=list(self.target_relation_types.values()),
            model=model,
            tokenizer=tokenizer,
        )

        predicted_relations = {predicted_relation}
        gt_relations = set(relation.type for relation in example.relations)
        correct_relations = gt_relations & predicted_relations

        return Counter({
            'num_sentences': 1,
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })


@register_dataset
class TACRED(RelationClassificationDataset):
    name = 'tacred'
    default_input_format = 'rel_input'
    default_output_format = 'rel_output'

    NO_RELATION = 'no relation'

    @staticmethod
    def to_natural(t: str) -> str:
        """
        Convert entity or relation type to a natural text.
        """
        t = t.split(":")
        assert len(t) <= 2, "Unexpected format {}".format(t)
        t = t[1] if len(t) == 2 else t[0]
        t = t.lower()
        t = t.replace("_", " ")
        t = t.replace("/", " ")
        t = t.replace("stateorprovince", "state or province")

        return t

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'json/{split}.json')
        examples = []

        # we fill entity/relation types while parsing the data
        self.entity_types = {}
        self.relation_types = {}

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")
            i = 0
            for idx, obj in enumerate(data):
                words = obj['token']
                head_start, head_end, head_type = obj['subj_start'], obj['subj_end'] + 1, obj['subj_type']
                tail_start, tail_end, tail_type = obj['obj_start'], obj['obj_end'] + 1, obj['obj_type']
                relation = obj['relation']

                if head_type not in self.entity_types:
                    self.entity_types[head_type] = EntityType(short=head_type, natural=self.to_natural(head_type))
                if tail_type not in self.entity_types:
                    self.entity_types[tail_type] = EntityType(short=tail_type, natural=self.to_natural(tail_type))

                head_entity = Entity(
                    id=None,
                    type=self.entity_types[head_type],
                    start=head_start,
                    end=head_end
                )
                tail_entity = Entity(
                    id=None,
                    type=self.entity_types[tail_type],
                    start=tail_start,
                    end=tail_end
                )

                entities = [
                    head_entity, tail_entity
                ]

                if relation not in self.relation_types:
                    self.relation_types[relation] = RelationType(short=relation, natural=self.to_natural(relation))

                relations = [
                    Relation(type=self.relation_types[relation], head=head_entity, tail=tail_entity)
                ]

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=words,
                    entities=entities,
                    relations=relations,
                )
                i += 1
                examples.append(example)

        self.relation_types = {
            relation.type.short: relation.type
            for example in examples for relation in example.relations
        }

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None,
                         eval_nll=False) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.

        We use NLL inference only when the generated output sentence has an unrecognized relation.
        """
        if not eval_nll and not self.eval_nll:
            # evaluate by generating the output sentence
            predicted_entities, predicted_relations = self.output_format.run_inference(
                example,
                output_sentence,
                entity_types=self.entity_types,
                relation_types=self.relation_types,
            )

            predicted_relation_str = next(iter(predicted_relations))[0]

            if predicted_relation_str not in [relation_type.natural for relation_type in self.relation_types.values()]:
                # the output relation is not in the list of possible relations, so we use NLL evaluation instead
                return self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=tokenizer,
                    eval_nll=True
                )

        else:
            # NLL evaluation
            predicted_relation_type = self.nll_inference(
                example=example,
                relation_types=list(self.relation_types.values()),
                model=model,
                tokenizer=tokenizer,
            )

            predicted_relation_str = predicted_relation_type.natural

        # load ground truth relation
        gt_relation_str = example.relations[0].type.natural

        if gt_relation_str == self.NO_RELATION and predicted_relation_str == self.NO_RELATION:
            return Counter({
                'num_sentences': 0,
                'gt_relations': 0,
                'predicted_relations': 0,
                'correct_relations': 0,
            })
        elif gt_relation_str == self.NO_RELATION:
            return Counter({
                'num_sentences': 1,
                'gt_relations': 0,
                'predicted_relations': 1,
                'correct_relations': 0,
            })
        elif predicted_relation_str == self.NO_RELATION:
            return Counter({
                'num_sentences': 1,
                'gt_relations': 1,
                'predicted_relations': 0,
                'correct_relations': 0,
            })
        else:
            return Counter({
                'num_sentences': 1,
                'gt_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 1 if predicted_relation_str == gt_relation_str else 0,
            })

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, and return relation metrics only.
        """
        results = super().evaluate_dataset(data_args, model, device, batch_size, macro=macro)
        return {k: v for k, v in results.items() if k.startswith('relation')}


@register_dataset
class CONLL05SRL(NERDataset):
    name = 'CoNLL2005-SRL'
    natural_entity_types = {
        'ARG0': 'first argument',
        'ARG1': 'second argument',
        'V': 'predicate',
    }

    default_input_format = 'srl_input'

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        predicate = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                current_entity = Entity(
                    type=EntityType(
                        short=tag_type,
                        natural=self.natural_entity_types[tag_type]
                        if tag_type in self.natural_entity_types else tag_type
                    ),
                    start=ii,
                    end=ii+1,
                )
                if tag_type == 'V':
                    predicate = current_entity
                else:
                    entities.append(current_entity)
            elif el.startswith('I-'):
                current_entity.end = ii + 1
        return entities, predicate
  
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'conll05.{split}.txt')
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('|||')
                assert len(line) == 2
                sentence, tag = line[0].strip(), line[1].strip()
                sentence = sentence.split()[1:]
                tag = tag.split()
                arguments, predicate = self.convert_bio_to_entities(tag)
                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=sentence,
                    entities=arguments,
                    relations=[],
                    sentence_level_entities=[predicate]
                )
                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }
        return examples


@register_dataset
class CONLL12SRL(NERDataset):
    name = 'CoNLL2012-SRL'
    natural_entity_types = {
        'ARGM-LOC': 'location'
    }

    default_input_format = 'srl_input'

    @staticmethod
    def get_word_idx(text: str) -> List[int]:
        word_idx = []
        for ii, c in enumerate(text):
            if len(word_idx) == 0:
                word_idx.append(0)
            elif c == ' ':
                word_idx.append(-1)
            elif word_idx[ii - 1] == -1:
                word_idx.append(word_idx[ii - 2] + 1)
            else:
                word_idx.append(word_idx[ii - 1])
        return word_idx
  
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.json')
        i = 0
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                text = obj['raw_text']
                text_split = text.split()
                char_to_word_idx = self.get_word_idx(text)

                for event in obj['events']:
                    # trigger
                    start_char, end_char = event['trigger']['start'], event['trigger']['end']
                    start_word, end_word = char_to_word_idx[start_char], char_to_word_idx[end_char - 1] + 1
                    predicate = Entity(
                        id=None, type=EntityType(short='V', natural='predicate'),
                        start=start_word, end=end_word
                    )

                    arguments = []
                    for arg in event['arguments']:
                        assert len(arg['values']) == 1
                        values = arg['values'][0]
                        arg_start_char, arg_end_char = values['start'], values['end']
                        arg_start_word, arg_end_word = char_to_word_idx[arg_start_char], \
                            char_to_word_idx[arg_end_char - 1] + 1
                        arg_name = arg['name']
                        argument = Entity(
                            id=None,
                            type=EntityType(short=arg_name, natural=self.natural_entity_types[arg_name]
                                            if arg_name in self.natural_entity_types else arg_name),
                            start=arg_start_word,
                            end=arg_end_word
                        )

                        arguments.append(argument)

                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=text_split,
                        entities=arguments,
                        relations=[],
                        sentence_level_entities=[predicate]
                    )
                    examples.append(example)
                    i += 1

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }
        return examples


@register_dataset
class CONLL05SRLBrown(CONLL05SRL):
    name = 'CoNLL2005-SRL-Brown'


@register_dataset
class MultiWoz(BaseDataset):
    """
    MultiWoz 2.1 dataset (Dialogue State Tracking).

    To obtain the multi-woz 2.1 dataset in TANL format:
        1. download the pre-processing script from trade-dst
            - git clone https://github.com/jasonwu0731/trade-dst.git
        2. update the dataset_url in file trade-dst/create_data.py line 309 to the url given below.
           This step ensures you are using the MultiWoz 2.1 dataset vs. the 2.0 version.
            - "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y"
        3. on lines 320, 321, 322, and 323 change the multi-woz directory to the following:
            - "data/multi-woz/MULTIWOZ2.1/"
        4. run the trade-dst pre-processing script:
            - python create_data.py
        5. copy the script ./trade-dst/utils/fix_label.py to tanl/preprocess_multiwoz
            - cp ./trade-dst/utils/fix_label.py tanl/preprocess_multiwoz
        6. run prepare_multi_woz.py
            - python prepare_multi_woz.py --data-dir ./trade-dst/data
        7. move the saved splits to ./data/multi_woz
            - mv ./trade-dst/data/splits ./data/multi_woz_2.1
    """
    name = 'multi_woz'
    data_name = 'multi_woz_2.1'
    default_input_format = 'plain'
    default_output_format = 'multi_woz'

    def truncate_first_n_tokens(self, examples, max_seq_length, delimiter='▁'):
        output = []
        for x in examples:
            tokens = self.tokenizer.tokenize(x)
            if len(tokens) > max_seq_length:
                x = ''.join(tokens[-1 * max_seq_length + 1:]).replace(delimiter, ' ')
                assert self.tokenizer.tokenize(''.join(tokens).replace(delimiter, ' ')) == tokens
            output.append(x)
        return output

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        file_path = os.path.join(self.data_dir(), f'multi_woz_2.1_{split}_5_domain.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            num_examples = len(data["examples"])
            logging.info(f"Loaded {num_examples} sentences for split {split} of {self.name}")

            for i, x in enumerate(data["examples"]):
                turn_id = x["turn_id"]
                conv_id = x["ID"]
                dialog = x["dialog_history"]
                tokens = dialog.split(" ")
                belief = x["turn_belief"]
                uid = "{0}-{1}".format(turn_id, conv_id)

                example = InputExample(
                    id=uid,
                    tokens=tokens,
                    belief_state=belief,
                    utterance_tokens=x["turn_uttr"],
                )
                examples.append(example)
        return examples

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        output_sentences = [self.output_format.format_output(example) for example in self.examples]

        input_sentences = self.truncate_first_n_tokens(examples=input_sentences,
                                                       max_seq_length=max_input_length)
        output_sentences = self.truncate_first_n_tokens(examples=output_sentences,
                                                        max_seq_length=max_output_length)

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0), print(
            f'Size does not match: len(sentences_tok.input_ids)={len(input_tok.input_ids)}, '
            f'len(labels_tok.input_ids)={len(output_tok.input_ids)}'
        )
        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids):
            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_input_ids.tolist()
            ))

        return features

    def evaluate(self, example: InputExample, output_sentence: str):
        """
        Evaluate a single example.
        """
        gold_belief_set = set(example.belief_state)
        pred_belief_set = \
            self.output_format.run_inference(
                example,
                output_sentence,
            )

        # remove "-"
        pred_belief_set = {elem.replace("-", " ") for elem in pred_belief_set}
        gold_belief_set = {elem.replace("-", " ") for elem in gold_belief_set}
        correct_belief_state = gold_belief_set == pred_belief_set

        return {
            'num_sentences': 1,
            'correct_state': int(correct_belief_state),
            'raw_gold_state': None,
            'raw_pred_state': output_sentence,
            'list_pred_state': list(pred_belief_set),
            'list_gold_state': list(gold_belief_set)
        }

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """

        def compute_accuracy(results_dict):
            num_examples = float(sum(results_dict["num_sentences"]))
            return sum(results_dict["correct_state"]) / num_examples

        results = {
            'num_sentences': [],
            'correct_state': [],
            'raw_gold_state': [],
            'raw_pred_state': [],
            'list_pred_state': [],
            'list_gold_state': []
        }

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate(
                example=example,
                output_sentence=output_sentence,
            )

            for k, v in new_result.items():
                results[k].append(v)

        return {
            'joint_accuracy': compute_accuracy(results),
        }
