import collections
import typing

import numpy as np

import metrics
import model


PredictionAndOriginal = typing.Tuple[metrics.LabeledSample, model.Sample]


def bin_predictions_by(samples: typing.Iterable[PredictionAndOriginal],
                       predicate: typing.Callable[[PredictionAndOriginal], float],
                       num_bins: int = 10,
                       threshold: typing.Optional[int] = None
                       ) -> typing.Tuple[typing.Dict[int, typing.Iterable[PredictionAndOriginal]], typing.List[float]]:
    samples = list(samples)
    values = []
    for s in samples:
        try:
            values.append(predicate(s))
        except ValueError:
            continue
    np_bins = np.linspace(np.min(values), np.max(values) + 1, num_bins + 1)
    bin_indices = np.digitize(values, np_bins)
    bins: typing.List[float] = np_bins.tolist()
    ret = collections.defaultdict(list)

    for i, bin_id in enumerate(bin_indices):
        bin_id = bin_id.item()
        ret[bin_id].append(samples[i])
        if bin_id == 0:
            print(f'left edge: {bins[0]}, value: {values[i]}')

    assert len(ret) <= num_bins
    assert 0 not in ret
    assert num_bins + 1 not in ret

    if threshold is not None:
        first_non_threshold_bin: typing.Optional[int] = None
        ret_items = list(ret.items())
        ret_items.sort(key=lambda x: x[0])
        for bin_id, bin_samples in ret_items:
            bin_count = len(bin_samples)
            if bin_count < threshold:
                if first_non_threshold_bin is None:
                    first_non_threshold_bin = bin_id
            else:
                first_non_threshold_bin = None

        if first_non_threshold_bin is not None:
            tmp = {}
            for bin_id, bin_samples in ret.items():
                if bin_id < first_non_threshold_bin:
                    tmp[bin_id] = bin_samples

            return tmp, bins[:first_non_threshold_bin]

    return ret, bins
