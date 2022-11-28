import abc

import typing

import model


class LocalSanityCheckError(Exception):
    def __init__(self, message: str, sample: model.Sample):
        super().__init__(message)
        self.sample = sample


class GlobalSanityCheckError(Exception):
    def __init__(self, message: str, samples: typing.List[model.Sample]):
        super().__init__(message)
        self.samples = samples


class BaseGlobalSanityCheck(abc.ABC):
    def holds(self, dataset: model.DataSet) -> None:
        raise NotImplementedError()


class BaseLocalSanityCheck(abc.ABC):
    def holds(self, sample: model.Sample) -> None:
        raise NotImplementedError()


class TokenBoundaryCheck(BaseLocalSanityCheck):
    """
    Checks if the text of tokens matches the text  that is defined by
    the span (start:stop) in the original sample text.
    """

    def holds(self, sample: model.Sample) -> None:
        for token in sample.tokens:
            is_text_equal = sample.text[token.start_char_index: token.stop_char_index] == token.text
            if not is_text_equal:
                raise LocalSanityCheckError(message=f'Expected token text "{token.text}" to be the same as '
                                                    f'the text marked by start ({token.start_char_index}) '
                                                    f'and stop ({token.stop_char_index}) char indices '
                                                    f'in the sample text, which is '
                                                    f'"{sample.text[token.start_char_index:token.stop_char_index]}" '
                                                    f'instead.',
                                            sample=sample)


class EntityFilledCheck(BaseLocalSanityCheck):
    """
    Checks if the entities in relations and the entities list are equal
    """

    def holds(self, sample: model.Sample) -> None:
        for r in sample.relations:
            if r.head not in sample.entities:
                raise LocalSanityCheckError(f'Head ({r.head}) does not appear in list of entities {sample.entities}.', sample)
            if r.tail not in sample.entities:
                raise LocalSanityCheckError(f'Tail ({r.tail}) does not appear in list of entities {sample.entities}', sample)


class TokenIndicesCheck(BaseLocalSanityCheck):
    """
    Checks if the text of entities matches the text that would result from concatenating
    the text of all tokens that make up the entity. This will also catch errors in the
    list of token indices of entities.
    """

    def __init__(self, case_sensitive: bool = False):
        self._case_sensitive = case_sensitive

    def holds(self, sample: model.Sample) -> None:
        for entity in sample.entities:
            reconstructed_text = ''
            last_stop: typing.Optional[int] = None
            for t_index in entity.token_indices:
                token = sample.tokens[t_index]
                if last_stop is not None:
                    reconstructed_text += ' ' * (token.start_char_index - last_stop)
                reconstructed_text += token.text
                last_stop = token.stop_char_index

            entity_text = entity.text
            if not self._case_sensitive:
                entity_text = entity_text.lower()
            for t_index in entity.token_indices:
                token = sample.tokens[t_index]
                token_text = token.text.lower()
                if not entity_text.startswith(token_text):
                    raise LocalSanityCheckError(message=f'Expected token with text "{token_text}" to appear '
                                                        f'in entity with text "{entity.text}", but it did not.',
                                                sample=sample)
                entity_text = entity_text[len(token.text):]
                entity_text = entity_text.strip()
            if len(entity_text) != 0:
                raise LocalSanityCheckError(message=f'Expected entity text "{entity.text}", '
                                                    f'to be made up of corresponding token texts '
                                                    f'{[sample.tokens[tid].text for tid in entity.token_indices]}.',
                                            sample=sample)
