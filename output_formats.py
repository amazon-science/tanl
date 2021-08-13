# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np

from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, CorefDocument
from utils import augment_sentence, get_span


OUTPUT_FORMATS = {}


def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseOutputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError

    def parse_output_sentence(self, example: InputExample, output_sentence: str) -> Tuple[list, bool]:
        """
        Parse an output sentence in augmented language and extract inferred entities and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        ['Tolkien', 'wrote', 'The', 'Lord', 'of', 'the', 'Rings']

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
        output_tokens = []
        unmatched_predicted_entities = []

        # add spaces around special tokens, so that they are alone when we split
        padded_output_sentence = output_sentence
        for special_token in [
            self.BEGIN_ENTITY_TOKEN, self.END_ENTITY_TOKEN,
            self.SEPARATOR_TOKEN, self.RELATION_SEPARATOR_TOKEN,
        ]:
            padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

        entity_stack = []   # stack of the entities we are extracting from the output sentence
        # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
        # where state is "name" (before the first | separator) or "other" (after the first | separator)

        for token in padded_output_sentence.split():
            if len(token) == 0:
                continue

            elif token == self.BEGIN_ENTITY_TOKEN:
                # begin entity
                start = len(output_tokens)
                entity_stack.append([start, "name", [], []])

            elif token == self.END_ENTITY_TOKEN and len(entity_stack) > 0:
                # end entity
                start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

                entity_name = ' '.join(entity_name_tokens).strip()
                end = len(output_tokens)

                tags = []

                # split entity_other_tokens by |
                splits = [
                    list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == self.SEPARATOR_TOKEN)
                    if not x
                ]

                if state == "other" and len(splits) > 0:
                    for x in splits:
                        tags.append(tuple(' '.join(x).split(' ' + self.RELATION_SEPARATOR_TOKEN + ' ')))

                unmatched_predicted_entities.append((entity_name, tags, start, end))

            else:
                # a normal token
                if len(entity_stack) > 0:
                    # inside some entities
                    if token == self.SEPARATOR_TOKEN:
                        x = entity_stack[-1]

                        if x[1] == "name":
                            # this token marks the end of name tokens for the current entity
                            x[1] = "other"
                        else:
                            # simply add this token to entity_other_tokens
                            x[3].append(token)

                    else:
                        is_name_token = True

                        for x in reversed(entity_stack):
                            # check state
                            if x[1] == "name":
                                # add this token to entity_name_tokens
                                x[2].append(token)

                            else:
                                # add this token to entity_other tokens and then stop going up in the tree
                                x[3].append(token)
                                is_name_token = False
                                break

                        if is_name_token:
                            output_tokens.append(token)

                else:
                    # outside
                    output_tokens.append(token)

        # check if we reconstructed the original sentence correctly, after removing all spaces
        wrong_reconstruction = (''.join(output_tokens) != ''.join(example.tokens))

        # now we align self.tokens with output_tokens (with dynamic programming)
        cost = np.zeros((len(example.tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
        # and output_tokens[:j]
        best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

        for i in range(len(example.tokens) + 1):
            for j in range(len(output_tokens) + 1):
                if i == 0 and j == 0:
                    continue

                candidates = []

                # match
                if i > 0 and j > 0:
                    candidates.append(
                        ((0 if example.tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

                # skip in the first sequence
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))

                # skip in the second sequence
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))

                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        # reconstruct best alignment
        matching = {}

        i = len(example.tokens) - 1
        j = len(output_tokens) - 1

        while i >= 0 and j >= 0:
            chosen_option = best[i + 1, j + 1]

            if chosen_option == 1:
                # match
                matching[j] = i
                i, j = i - 1, j - 1

            elif chosen_option == 2:
                # skip in the first sequence
                i -= 1

            else:
                # skip in the second sequence
                j -= 1

        # update predicted entities with the positions in the original sentence
        predicted_entities = []

        for entity_name, entity_tags, start, end in unmatched_predicted_entities:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            for j in range(start, end):
                if j in matching:
                    if new_start is None:
                        new_start = matching[j]

                    new_end = matching[j]

            if new_start is not None:
                # predict entity
                entity_tuple = (entity_name, entity_tags, new_start, new_end + 1)
                predicted_entities.append(entity_tuple)

        return predicted_entities, wrong_reconstruction


@register_output_format
class JointEROutputFormat(BaseOutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 'joint_er'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [ Tolkien | person | born in = here ] was born [ here | location ]
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.entities}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool, bool, bool, bool]:
        """
        Process an output sentence to extract predicted entities and relations (among the given entity/relation types).

        Return the predicted entities, predicted relations, and four booleans which describe if certain kinds of errors
        occurred (wrong reconstruction of the sentence, label error, entity error, augmented language format error).
        """
        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        entity_error = False    # whether there is at least one relation pointing to a non-existing head entity
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()
        raw_predicted_relations = []

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
                predicted_entities_by_name[entity_name].append(entity_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            raw_predicted_relations.append((relation_type, entity_tuple, related_entity))
                        else:
                            label_error = True

                    else:
                        # the relation tag has the wrong length
                        format_error = True

            else:
                # the predicted entity type does not exist
                label_error = True

        predicted_relations = set()

        for relation_type, entity_tuple, related_entity in raw_predicted_relations:
            if related_entity in predicted_entities_by_name:
                # look for the closest instance of the related entity
                # (there could be many of them)
                _, head_start, head_end = entity_tuple
                candidates = sorted(
                    predicted_entities_by_name[related_entity],
                    key=lambda x:
                    min(abs(x[1] - head_end), abs(head_start - x[2]))
                )

                for candidate in candidates:
                    relation = (relation_type, entity_tuple, candidate)

                    if relation not in predicted_relations:
                        predicted_relations.add(relation)
                        break

            else:
                # cannot find the related entity in the sentence
                entity_error = True

        return predicted_entities, predicted_relations, wrong_reconstruction, label_error, entity_error, format_error


@register_output_format
class JointICSLFormat(JointEROutputFormat):
    """
    Output format for joint intent classification and slot labeling.
    """
    name = 'joint_icsl'
    BEGIN_INTENT_TOKEN = "(("
    END_INTENT_TOKEN = "))"
        
    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language.
        """
        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        augmented_sentence = augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)
        
        return (f"(( {example.intent.natural} )) " + augmented_sentence)

    def run_inference(self, example: InputExample, output_sentence: str,
            entity_types: Dict[str, EntityType] = None) -> Tuple[str, set]:
        entity_types = set(entity_type.natural for entity_type in entity_types.values())

        # parse output sentence
        # get intent
        for special_token in [self.BEGIN_INTENT_TOKEN, self.END_INTENT_TOKEN]:
            output_sentence.replace(special_token, ' ' + special_token + ' ')

        output_sentence_tokens = output_sentence.split()

        if self.BEGIN_INTENT_TOKEN in output_sentence_tokens and \
                self.END_INTENT_TOKEN in output_sentence_tokens:
            intent = output_sentence.split(self.BEGIN_INTENT_TOKEN)[1].split(self.END_INTENT_TOKEN)[0].strip()
            output_sentence = output_sentence.split(self.END_INTENT_TOKEN)[1]   # remove intent from sentence

        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
            else:
                label_error = True

        return intent, predicted_entities, wrong_reconstruction, label_error, format_error
  

@register_output_format
class EventOutputFormat(JointEROutputFormat):
    """
    Output format for event extraction, where an input example contains exactly one trigger.
    """
    name = 'ace2005_event'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, similarly to JointEROutputFormat (but we also consider triggers).
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.entities + example.triggers}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in (example.entities + example.triggers):
            if not relations_by_entity[entity]:
                continue

            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool]:
        """
        Process an output sentence to extract arguments, given as entities and relations.
        """
        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}

        triggers = example.triggers
        assert len(triggers) <= 1
        if len(triggers) == 0:
            # we do not have triggers
            return set(), set(), False

        trigger = triggers[0]

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities = set()
        predicted_relations = set()

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            predicted_relations.add(
                                (relation_type, entity_tuple, (trigger.type.natural, trigger.start, trigger.end))
                            )

        return predicted_entities, predicted_relations, wrong_reconstruction


@register_output_format
class CorefOutputFormat(BaseOutputFormat):
    """
    Output format for coreference resolution.
    """
    name = 'coref'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        Tolkien's epic novel [ The Lord of the Rings ] was published in 1954-1955, years after the
        [ book | The Lord of the Rings ] was completed.
        """
        augmentations = []

        for group in example.groups:
            previous_entity = None
            for entity in group:
                augmentation = (
                    [(' '.join(example.tokens[previous_entity.start:previous_entity.end]),)]
                    if previous_entity is not None else [],
                    entity.start,
                    entity.end,
                )
                augmentations.append(augmentation)
                previous_entity = entity

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str) \
            -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Process an output sentence to extract coreference relations.

        Return a list of ((start, end), parent) where (start, end) denote an entity span, and parent is either None
        or another (previous) entity span.
        """
        raw_annotations, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        res = []
        previous_entities = {}
        for entity, tags, start, end in raw_annotations:
            entity_span = (start, end)

            if len(tags) > 0 and tags[0][0] in previous_entities:
                previous_entity = tags[0][0]
                res.append((entity_span, previous_entities[previous_entity]))

            else:
                # no previous entity found
                res.append((entity_span, None))

            # record this entity
            previous_entities[entity] = entity_span

        return res


@register_output_format
class RelationClassificationOutputFormat(BaseOutputFormat):
    """
    Output format for relation classification.
    """
    name = 'rel_output'

    def format_output(self, example: InputExample) -> str:
        en1_span = [example.entities[0].start, example.entities[0].end]
        en2_span = [example.entities[1].start, example.entities[1].end]
        words = example.tokens
        s = f"relationship between {self.BEGIN_ENTITY_TOKEN} {get_span(words, en1_span)} {self.END_ENTITY_TOKEN} and " \
            f"{self.BEGIN_ENTITY_TOKEN} {get_span(words, en2_span)} {self.END_ENTITY_TOKEN} " \
            f"{self.RELATION_SEPARATOR_TOKEN} {example.relations[0].type.natural}"
        return s.strip()

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set]:
        """
        Process an output sentence to extract the predicted relation.

        Return an empty list of entities and a single relation, so that it is compatible with joint entity-relation
        extraction datasets.
        """
        predicted_relation = output_sentence.split(self.RELATION_SEPARATOR_TOKEN)[-1].strip()
        predicted_entities = set()  # leave this empty as we only predict the relation

        predicted_relations = {(
            predicted_relation,
            example.relations[0].head.to_tuple() if example.relations[0].head else None,
            example.relations[0].tail.to_tuple() if example.relations[0].tail else None,
        )}

        return predicted_entities, predicted_relations


@register_output_format
class MultiWozOutputFormat(BaseOutputFormat):
    """
    Output format for the MultiWoz DST dataset.
    """
    name = 'multi_woz'

    none_slot_value = 'not given'
    domain_ontology = {
        'hotel': [
            'price range',
            'type',
            'parking',
            'book stay',
            'book day',
            'book people',
            'area',
            'stars',
            'internet',
            'name'
        ],
        'train': [
            'destination',
            'day',
            'departure',
            'arrive by',
            'book people',
            'leave at'
        ],
        'attraction': ['type', 'area', 'name'],
        'restaurant': [
            'book people',
            'book day',
            'book time',
            'food',
            'price range',
            'name',
            'area'
        ],
        'taxi': ['leave at', 'destination', 'departure', 'arrive by'],
        'bus': ['people', 'leave at', 'destination', 'day', 'arrive by', 'departure'],
        'hospital': ['department']
    }

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [belief] hotel price range cheap , hotel type hotel , duration two [belief]
        """
        turn_belief = example.belief_state
        domain_to_slots = defaultdict(dict)
        for label in turn_belief:
            domain, slot, value = label.split("-")
            domain_to_slots[domain][slot] = value

        # add slots that are not given
        for domain, slot_dict in domain_to_slots.items():
            for slot in self.domain_ontology[domain]:
                if slot not in slot_dict:
                    slot_dict[slot] = self.none_slot_value

        output_list = []
        for domain, slot_dict in sorted(domain_to_slots.items(), key=lambda p: p[0]):
            output_list += [
                f"{domain} {slot} {value}" for slot, value in sorted(slot_dict.items(), key=lambda p: p[0])
            ]
        output = " , ".join(output_list)
        output = f"[belief] {output} [belief]"
        return output

    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract the predicted belief.
        """
        start = output_sentence.find("[belief]")
        end = output_sentence.rfind("[belief]")

        label_span = output_sentence[start+len("[belief]"):end]
        belief_set = set([
            slot_value.strip() for slot_value in label_span.split(",")
            if self.none_slot_value not in slot_value
        ])
        return belief_set
