import pyterrier as pt
from pyterrier.measures import *
from pyterrier_dr import NumpyIndex, TctColBert
import pandas as pd
import sys
import os

pt.init()

# Dense Retrieval
model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
index_dr = NumpyIndex(f'indices/{sys.argv[1]}_tct_colbert_v2_hnp_msmarco.np')

dr = model >> index_dr

# BM25
index_bm25 = pt.IndexFactory.of(f"/home/obez/InformationRetrieval/indices/{sys.argv[1]}_bm25_index")

bm25 = pt.BatchRetrieve(
    index_bm25,
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": ""},
    controls={"qe": "off"},
)

print("The indexes are loaded")

if sys.argv[1] == "arguana":
    dataset = pt.get_dataset('./arguana-passage-beir')
elif sys.argv[1] == "msmarco_passages":
    dataset = pt.get_dataset('irds:msmarco-passage/dev/small')

print(f"The dataset used is: {dataset.info_url()}")

results = pt.Experiment(
    [dr, bm25],
    dataset.get_topics()[:500],
    dataset.get_qrels(),
    eval_metrics=[nDCG@10, RR@10, R@10, R@100, P@10, P@100, SetP],
    filter_by_topics=True,
    dataframe=True,
)

if not os.path.exists("./baseline/"):
    os.makedirs("./baseline/")

results.to_csv(f"./baseline/{sys.argv[1]}.csv")

print(f"Saved results .csv in: ./baseline/{sys.argv[1]}.csv")

# RR - Reciprocal Rank, absolute ordering
# nDCG - ordering again
# R - recall - important metric for retrieval - relevant docs retrieved/ all relevant 
# P - precision - relevant documents retrieved/ all retrieved
# SetP  - Set precision - relecant docs retrieved/ all relevant + all retrieved (Jaccard)