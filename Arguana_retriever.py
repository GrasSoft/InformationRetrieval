import pyterrier as pt
from pyterrier.measures import *
from pyterrier_dr import NumpyIndex, TctColBert
import pandas as pd

pt.init()

model = TctColBert('castorini/tct_colbert-v2-msmarco')
index = NumpyIndex('indices/arguana_tct_colbert_v2_msmarco.np')

retrieval_pipeline = model >> index

query = "African"

query_df = pd.DataFrame([{"qid": "1", "query": query}])

results = retrieval_pipeline.transform(query_df)

print("Dense Retriever")
print(results.head(10))

index_ref = pt.IndexFactory.of("/home/obez/InformationRetrieval/indices/arguana_bm25_index")

bm25 = pt.BatchRetrieve(
    index_ref,
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": ""},
    controls={"qe": "off"},
)


query_df = pd.DataFrame([{"qid": "1", "query": query}])
print("BM25")
print((bm25 % 10).transform(query_df))
