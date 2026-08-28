from typing import TypedDict

BASE_URL = 'https://www.ncbi.nlm.nih.gov/research/pubtator3-api'


class PubtatorAnnotation(TypedDict):
    pmid: str
    pmcid: str
    text: str
    prefix: str
    suffix: str
