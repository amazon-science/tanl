# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2017 Kenton Lee
# SPDX-License-Identifier: Apache-2.0

# uses some code from
# https://github.com/kentonl/e2e-coref/blob/master/metrics.py


from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


MUC = 'muc'
BCUBED = 'b_cubed'
CEAFE = 'ceafe'


class CorefAllMetrics(object):
    """
    Wrapper for coreference resolution metrics.
    """

    @staticmethod
    def _get_mention_to_x(clusters: List[list]) -> dict:
        mention_to_x = {}
        for cluster in clusters:
            for m in cluster:
                mention_to_x[m] = tuple(cluster)
        return mention_to_x

    def _compute_coref_metrics(self, gold_clusters: List[list], predicted_clusters: List[list]) \
            -> Dict[str, Dict[str, float]]:
        """
        Compute all coreference metrics given a list of gold cluster and a list of predicted clusters.
        """
        mention_to_predicted = self._get_mention_to_x(predicted_clusters)
        mention_to_gold = self._get_mention_to_x(gold_clusters)
        result = {}
        metric_name_evals = [('muc', Evaluator(muc)), ('b_cubed', Evaluator(b_cubed)), ('ceaf', Evaluator(ceafe))]

        for name, evaluator in metric_name_evals:
            evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
            result[name] = {
                'precision': evaluator.get_precision(),
                'recall': evaluator.get_recall(),
                'f1': evaluator.get_f1()
            }

        result['average'] = {
            'precision': sum([result[k]['precision'] for k, _ in metric_name_evals]) / len(metric_name_evals),
            'recall': sum([result[k]['recall'] for k, _ in metric_name_evals]) / len(metric_name_evals),
            'f1': sum([result[k]['f1'] for k, _ in metric_name_evals]) / len(metric_name_evals)
        }

        return result

    @staticmethod
    def _average_nested_dict(list_nested_dict: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Given a list of 2-level nested dict, compute the average.
        """
        result_dict = {}

        # sum up all values
        for outer_dict in list_nested_dict:
            for key_outer, value_outer in outer_dict.items():
                if key_outer not in result_dict:
                    result_dict[key_outer] = {}
                for key_inner, value_inner in value_outer.items():
                    result_dict[key_outer][key_inner] = result_dict[key_outer].get(key_inner, 0.0) + value_inner

        # take the average
        for key_outer, value_outer in result_dict.items():
            for key_inner, value_inner in value_outer.items():
                result_dict[key_outer][key_inner] = result_dict[key_outer][key_inner] / len(list_nested_dict)

        return result_dict

    def get_all_metrics(self, labels: List[List[List[Tuple[int, int]]]], preds: List[List[List[Tuple[int, int]]]])\
            -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute all metrics for coreference resolution.

        In input are given two list of mention groups, for example:
        [   # this is the corpus level, with a list of documents
            [   # this is the document level, with a list of mention clusters
                [   # this is the cluster level, with a list of spans
                    (5, 7),
                    (11, 19),
                    ...
                ],
                ...
            ]
        ]
        """
        assert len(labels) == len(preds)
        result = {}

        # compute micro-averaged scores (treat all clusters from all docs as a single list of clusters)
        gold_clusters = [
            [(i,) + span for span in cluster] for i, clusters in enumerate(labels) for cluster in clusters
        ]
        predicted_clusters = [
            [(i,) + span for span in cluster] for i, clusters in enumerate(preds) for cluster in clusters
        ]

        result['micro'] = self._compute_coref_metrics(gold_clusters, predicted_clusters)

        # compute macro-averaged scores (compute p/r/f1 for each doc first, then take average per doc)
        doc_metrics = []
        for gold_clusters, predicted_clusters in zip(labels, preds):
            doc_metrics.append(self._compute_coref_metrics(
                gold_clusters, predicted_clusters
            ))
        result['macro'] = self._average_nested_dict(doc_metrics)

        return result


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.metric_names = [MUC, BCUBED, CEAFE]
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]
        assert len(self.evaluators) == len(self.metric_names)
        self.name_to_evaluator = {n: e for n, e in zip(self.metric_names, self.evaluators)}

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold, mention_to_predicted, mention_to_gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)

        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:  # loop over each cluster
        gold_counts = Counter()
        correct = 0
        for m in c:     # loop over each mention
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count
        num += correct / float(len(c))
        dem += len(c)
    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(matrix1, matrix2):
    m_sum1 = np.sum(matrix1, axis=1)
    m_sum2 = np.sum(matrix2, axis=0)
    return 2 * np.dot(matrix1, matrix2) / (np.outer(m_sum1, np.ones_like(
        m_sum2)) + np.outer(np.ones_like(m_sum1), m_sum2))


def ceafe(clusters, gold_clusters, mention_to_predicted, mention_to_gold):
    key_list = list(set(mention_to_gold.keys()).union(
        set(mention_to_predicted.keys())))

    key_to_ix = {}
    for i, k in enumerate(key_list):
        key_to_ix[k] = i

    len_key = len(key_list)
    pred_matrix = np.zeros((len(clusters), len_key))
    gold_matrix = np.zeros((len(gold_clusters), len_key))
    fill_cluster_to_matrix(clusters, pred_matrix, key_to_ix)
    fill_cluster_to_matrix(gold_clusters, gold_matrix, key_to_ix)
    scores = phi4(pred_matrix, gold_matrix.transpose())
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()

    return similarity, len(clusters), similarity, len(gold_clusters)


def fill_cluster_to_matrix(clusters, matrix, key_to_ix):
    for i, c in enumerate(clusters):
        for m in c:
            matrix[i][key_to_ix[m]] = 1
