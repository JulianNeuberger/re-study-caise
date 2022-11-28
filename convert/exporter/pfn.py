import model
from convert.exporter import base


class PFNExporter(base.BaseExporter):
    def save(self, data_set: model.DataSet, file_path: str) -> None:
        pass