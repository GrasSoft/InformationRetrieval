import pyterrier as pt
from pyterrier.measures import *
from pyterrier_dr import NumpyIndex, TctColBert
import pandas as pd

pt.init()

model = TctColBert('castorini/tct_colbert-v2-msmarco')
index_ref = NumpyIndex('indices/msmarco_passages_tct_colbert_v2_msmarco.np')

dense_retrieval = model >> index_ref

index = pt.IndexFactory.of("/home/obez/InformationRetrieval/indices/msmarco_passages_bm25_index")

bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": ""},
    controls={"qe": "off"},
)

dataset = pt.get_dataset('irds:msmarco-passage/dev/small').get_qrels()


print(len(set(pt.get_dataset('irds:msmarco-passage/dev/small').get_qrels()["qid"])))

# pt.Experiment(
#     [dense_retrieval],
#     dataset.get_topics()[:2_000],
#     dataset.get_qrels(),
#     eval_metrics=[pt.measures.nDCG@10, pt.measures.RR@10],
# )