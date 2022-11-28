import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class DataSet:
    name: str
    samples: typing.List['Sample']


@dataclasses.dataclass(frozen=True)
class Sample:
    id: str
    text: str
    entities: typing.List['Entity']
    tokens: typing.List['Token']
    relations: typing.List['Relation']


@dataclasses.dataclass(frozen=True, eq=False)
class Entity:
    text: str
    ner_tag: str
    token_indices: typing.List[int]

    def __hash__(self) -> int:
        indices_as_tuple = (i for i in self.token_indices)
        return hash((self.text, self.ner_tag, indices_as_tuple))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Entity):
            return False
        if not self.text == o.text:
            return False
        if not self.ner_tag == o.ner_tag:
            return False
        if not self.token_indices == o.token_indices:
            return False
        return True


@dataclasses.dataclass(frozen=True)
class Token:
    text: str
    start_char_index: int
    stop_char_index: int
    ner_tag: typing.Optional[str]
    pos_tag: typing.Optional[str]
    dependency_relation: typing.Tuple[typing.Optional[int], str]


@dataclasses.dataclass(frozen=True)
class Relation:
    type: str
    head: Entity
    tail: Entity
