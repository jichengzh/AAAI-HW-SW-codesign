"""Expose copies of original rail samples after the frozen reader finishes."""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_PATH=Path(__file__).with_name('orin_power_reader_base.py')
BASE_SHA256='5ce12ed85bcf092db4ef3fe27bbd4d4f86547f6e5a11da826c9d9f813fb5e493'


class RawSamplesReader:
    def __init__(self,base):self._base=base

    def probe(self):return self._base.probe()

    def start_request(self):return self._base.start_request()

    def finish_request(self):
        result=self._base.finish_request()
        samples=[list(pair) for pair in self._base._samples]
        return {**result,'raw_power_samples':samples}


def open_reader():
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA256:
        raise ValueError('Frozen base reader SHA differs')
    manifest=json.loads(BASE_PATH.with_name('manifest.json').read_text())
    if manifest.get('power_reader_base_sha256')!=BASE_SHA256:
        raise ValueError('Frozen base reader manifest binding differs')
    spec=importlib.util.spec_from_file_location('_quant_original_power_reader',BASE_PATH)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    return RawSamplesReader(base.open_reader())
