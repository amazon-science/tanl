# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod
import copy

from input_example import InputExample
from utils import augment_sentence, get_span

INPUT_FORMATS = {}


def register_input_format(format_class):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseInputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='
    QUERY_SEPARATOR_TOKEN = ':'

    def format_input(self, example: InputExample, multitask=False, task_descriptor=None):
        res = self._format_input(example=example)
        if multitask:
            name = task_descriptor or example.dataset.task_descriptor or example.dataset.name
            res = f'{name} {self.QUERY_SEPARATOR_TOKEN} ' + res
        return res

    @abstractmethod
    def _format_input(self, example: InputExample) -> str:
        raise NotImplementedError


@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample) -> str:
        return ' '.join(example.tokens)


@register_input_format
class RelationClassificationInputFormat(BaseInputFormat):
    """
    Input format for relation classification.
    """
    name = 'rel_input'

    def _format_input(self, example: InputExample) -> str:
        en1_span = [example.entities[0].start, example.entities[0].end]
        en2_span = [example.entities[1].start, example.entities[1].end]
        words = example.tokens
        first, latter, head_first = (en1_span, en2_span, True) if en1_span[0] < en2_span[0] \
            else (en2_span, en1_span, False)

        s = " ".join(words[:first[0]]) \
            + f" {self.BEGIN_ENTITY_TOKEN} {get_span(words, first)} {self.END_ENTITY_TOKEN} " \
            + " ".join(words[first[1]:latter[0]])
        s += f" {self.BEGIN_ENTITY_TOKEN} {get_span(words, latter)} {self.END_ENTITY_TOKEN} " \
             + " ".join(words[latter[1]:])
        s += f" The relationship between {self.BEGIN_ENTITY_TOKEN} {get_span(words, en1_span)} " \
             f"{self.END_ENTITY_TOKEN} and {self.BEGIN_ENTITY_TOKEN} {get_span(words, en2_span)} " \
             f"{self.END_ENTITY_TOKEN} is"

        return s.strip()


@register_input_format
class EventInputFormat(BaseInputFormat):
    """
    Input format for event extraction, where an input example contains exactly one trigger.
    """
    name = 'ace2005_event_with_trigger'

    def _format_input(self, example: InputExample) -> str:
        triggers = example.triggers
        assert len(triggers) <= 1
        augmentations = [([(entity.type.natural,)], entity.start, entity.end) for entity in triggers]

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)


@register_input_format
class SRLInput(BaseInputFormat):
    """
    Input format for SRL, where the predicate is marked.
    """
    name = 'srl_input'

    def _format_input(self, example) -> str:
        assert len(example.sentence_level_entities) == 1
        start, end = example.sentence_level_entities[0].start, example.sentence_level_entities[0].end
        words = copy.copy(example.tokens)
        words.insert(end, self.END_ENTITY_TOKEN)
        words.insert(start, self.BEGIN_ENTITY_TOKEN)
        return ' '.join(words)
