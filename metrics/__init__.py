import collections
import dataclasses
import typing
from typing import Iterable

import numpy as np

from metrics.parser import LabeledSample, BaseParser, TwoAreBetterThanOneParser, JointERParser, CasRelParser, \
    RSANParser, SpertParser, PFNCustomParser, MareParser
from metrics.spans import BaseMatcher, ExactRelationMatcher, BoundaryRelationMatcher, EntityMatcher

available_result_parsers: typing.Dict[str, BaseParser] = {
    'two-are-better-than-one': TwoAreBetterThanOneParser(),
    'joint-er': JointERParser(),
    'casrel': CasRelParser(),
    'rsan': RSANParser(),
    'spert': SpertParser(),
    'pfn': PFNCustomParser(),
    'mare': MareParser()
}


@dataclasses.dataclass
class F1Metrics:
    f1: float
    precision: float
    recall: float


def p_score(n_gold: float, n_ok: float, n_pred: float) -> float:
    p = 0.0
    if n_pred == 0:
        if n_gold == 0:
            p = 1.0
    else:
        p = n_ok / n_pred
    return p


def r_score(n_gold: float, n_ok: float, n_pred: float) -> float:
    r = 0.0
    if n_gold == 0:
        if n_pred == 0:
            r = 1.0
    else:
        r = n_ok / n_gold
    return r


def f1_score(p: float, r: float) -> float:
    f1 = 0.0
    if p + r != 0.0:
        f1 = 2 * p * r / (p + r)
    return f1


def f1_micro(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    matches = [match_strategy.match(p) for p in predictions]
    n_true = sum([m.n_true for m in matches])
    n_pred = sum([m.n_pred for m in matches])
    n_ok = sum([m.n_ok for m in matches])

    p = p_score(n_true, n_ok, n_pred)
    r = r_score(n_true, n_ok, n_pred)
    f1 = f1_score(p, r)

    return F1Metrics(
        f1=f1,
        precision=p,
        recall=r
    )


def f1_macro_documents(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    matches = [match_strategy.match(p) for p in predictions]

    p_scores = []
    r_scores = []
    for m in matches:
        n_gold, n_ok, n_pred = m.n_true, m.n_ok, m.n_pred

        p = p_score(n_gold, n_ok, n_pred)
        p_scores.append(p)

        r = r_score(n_gold, n_ok, n_pred)
        r_scores.append(r)

    p = sum(p_scores) / len(p_scores)
    r = sum(r_scores) / len(r_scores)
    f1 = f1_score(p, r)

    return F1Metrics(
        f1=f1,
        precision=p,
        recall=r
    )


def f1_macro_relations(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    scores_by_relation_type: typing.Dict[str, typing.Tuple[float, float, float]] = collections.defaultdict(lambda: (0, 0, 0))

    for p in predictions:
        for rel in p.labels:
            # count the number of hits for this specific relation
            tmp = LabeledSample(labels=[rel], prediction=p.prediction, sample_id='')
            matches = match_strategy.match(tmp)

            n_gold, n_pred, n_ok = scores_by_relation_type[rel.tag]
            scores_by_relation_type[rel.tag] = (n_gold + matches.n_true, n_pred + matches.n_pred, n_ok + matches.n_ok)

    p_scores = []
    r_scores = []
    for rel_type, (n_gold, n_pred, n_ok) in scores_by_relation_type.items():
        p = p_score(n_gold, n_ok, n_pred)
        p_scores.append(p)

        r = r_score(n_gold, n_ok, n_pred)
        r_scores.append(r)

    p = sum(p_scores) / len(p_scores)
    r = sum(r_scores) / len(r_scores)
    f1 = f1_score(p, r)

    return F1Metrics(
        precision=p,
        recall=r,
        f1=f1
    )


def calculate_f1(predictions: Iterable[LabeledSample],
                 match_strategy: BaseMatcher,
                 f1_mode: str):
    assert f1_mode in ['micro', 'macro-documents', 'macro-relations']

    if f1_mode == 'macro-documents':
        return f1_macro_documents(predictions, match_strategy)

    if f1_mode == 'micro':
        return f1_micro(predictions, match_strategy)

    if f1_mode == 'macro-relations':
        return f1_macro_relations(predictions, match_strategy)


def calculate_confusion_matrix(predictions: Iterable[LabeledSample],
                               label_list: typing.List[str],
                               match_strategy: BoundaryRelationMatcher) -> typing.Tuple[np.ndarray, int, int, int]:
    confusion_matrix = np.zeros(shape=(len(label_list), len(label_list)))
    n_gold_entities = 0
    n_pred_entities = 0
    n_ok_entities = 0
    for p in predictions:
        match_result = match_strategy.match(p)
        matches = match_result.matches
        n_gold_entities += match_result.n_true
        n_pred_entities += match_result.n_pred
        n_ok_entities += match_result.n_ok
        for match in matches:
            confusion_matrix[label_list.index(match[0].tag), label_list.index(match[1].tag)] += 1

    return confusion_matrix, n_gold_entities, n_pred_entities, n_ok_entities
