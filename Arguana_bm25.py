import pyterrier as pt
from pathlib import Path
import pandas as pd

DATASET = pt.datasets.get_dataset("irds:beir/arguana")
INDEX = pt.index.IterDictIndexer(
    str(Path.cwd()),
    meta={
        "docno": 100,
        "text": 131072,
    },
    type=pt.index.IndexingType.MEMORY,
).index(DATASET.get_corpus_iter())


# bm25 = pt.terrier.Retriever(index, wmodel="BM25")

BM25 = pt.BatchRetrieve(
    INDEX,
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": ""},
    controls={"qe": "off"},
)


def search(query: str) -> pd.DataFrame:
    return (BM25 % 10).search(query)

print(search("African countries"))
