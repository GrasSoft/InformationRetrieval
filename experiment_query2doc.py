import pyterrier as pt
from pyterrier.measures import *
from pyterrier_dr import NumpyIndex, TctColBert
import pandas as pd
import sys
import os
import re

pt.init()

# Dense Retrieval
model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
index_dr = NumpyIndex(f'indices/trec_covid_tct_colbert_v2_hnp_msmarco.np')

dr = model >> index_dr

# BM25
index_bm25 = pt.IndexFactory.of(f"/home/obez/InformationRetrieval/indices/trec_covid_bm25_index")

bm25 = pt.BatchRetrieve(
    index_bm25,
    wmodel="BM25",
    metadata=["docno", "text"],
    properties={"termpipelines": "Stopwords,PorterStemmer"},
    controls={"qe": "off"},
)

print("The indexes are loaded")

dataset = pt.get_dataset("irds:beir/trec-covid")


topics = dataset.get_topics()
topics['qid'] = topics['qid'].astype(str)
qrels = dataset.get_qrels()
qrels['qid'] = qrels['qid'].astype(str)



print(f"The dataset used is: {dataset.info_url()}")

topics = pd.read_csv(f"./query_short/trec-covid-query2doc/trec-covid-query2doc.csv")[:500]
topics['qid'] = topics['qid'].astype(str)


def extract_numbered_items(text):
    # Regex pattern to match lines starting with a number followed by a dot and space
    pattern = r'\b\d+\.\s+(.*)'
    items = re.findall(pattern, text)
    return items

topics["query"] = topics["query"].apply(extract_numbered_items).apply(lambda x: ' '.join(x[:5]) if len(x) > 5 else ' '.join(x)).apply(lambda x: re.sub(r'\s+', ' ', x)).apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))
topics["query"] = topics["orig_query"] + " " + topics["query"]
topics.drop(columns=["orig_query"], inplace=True)

results = pt.Experiment(
    [bm25, dr],
    topics,
    qrels,
    eval_metrics=[nDCG@10, RR@10, R@10, R@100, P@10, P@100, SetP],
    filter_by_topics=True,
    dataframe=True,
)

if not os.path.exists(f"./modified/trec-covid-query2doc"):
    os.makedirs(f"./modified/trec-covid-query2doc")

results.to_csv(f"./modified/trec-covid-query2doc/trec-covid-query2doc_experiment.csv")

print(f"Saved results .csv in: ./modified/trec-covid-query2doc/trec-covid-query2doc_experiment.csv")
        

# RR - Reciprocal Rank, absolute ordering
# nDCG - ordering again
# R - recall - important metric for retrieval - relevant docs retrieved/ all relevant 
# P - precision - relevant documents retrieved/ all retrieved
# SetP  - Set precision - relecant docs retrieved/ all relevant + all retrieved (Jaccard)