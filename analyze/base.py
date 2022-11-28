import abc
import dataclasses
import typing

import model


@dataclasses.dataclass
class AnalysisResult:
    id: str
    value: typing.Any
    verbose_title: str


class AnalysisStep(abc.ABC):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        raise NotImplementedError()
