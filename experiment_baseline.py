import pyterrier as pt
from pyterrier.measures import *
from pyterrier_dr import NumpyIndex, TctColBert
import pandas as pd
import sys
import os

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

if sys.argv[1] == "arguana":
    dataset = pt.get_dataset('irds:beir/arguana')
    topics = dataset.get_topics()[:500]
elif sys.argv[1] == "msmarco_passages":
    dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
    topics = dataset.get_topics()[:500]
elif sys.argv[1] == "trec_covid":
    dataset = pt.get_dataset("irds:beir/trec-covid")
    topics = dataset.get_topics('query')[:500]
elif sys.argv[1] == "msmarco_passages_mis":
    dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
    topics = pd.read_csv("./query_misspelled_datasets/irds:msmarco-passagedevsmall.csv").drop(columns=['query']).rename(columns={'modified_query': 'query'})[:500]
elif sys.argv[1] == "arguana_mis":
    dataset = pt.get_dataset('irds:beir/arguana')
    topics = pd.read_csv("./query_misspelled_datasets/irds:beirarguana.csv").drop(columns=['query']).rename(columns={'modified_query': 'query'})[:500]    
elif sys.argv[1] == "trec_covid_mis":
    dataset = pt.get_dataset("irds:beir/trec-covid")
    topics = pd.read_csv("./query_misspelled_datasets/irds:beirtrec-covid.csv").drop(columns=['query']).rename(columns={'modified_query': 'query'})[:500]
elif sys.argv[1] == "trec-covid-misspelled":
    dataset = pt.get_dataset("irds:beir/trec-covid")
    topics = dataset.get_topics('query')

topics['qid'] = topics['qid'].astype(str)
qrels = dataset.get_qrels()
qrels['qid'] = qrels['qid'].astype(str)


    
bo1 = bm25%50 >> pt.rewrite.Bo1QueryExpansion(index_bm25) >> bm25
kl  = bm25%50 >> pt.rewrite.KLQueryExpansion(index_bm25)  >> bm25

print(f"The dataset used is: {dataset.info_url()}")



if len(sys.argv) > 2 and sys.argv[2] == "mod":
    for filename in os.listdir(f"./query_short/{sys.argv[1]}"):
        topics = pd.read_csv(f"./query_short/{sys.argv[1]}/{filename}").drop(columns=['orig_query'])[:500]
        print(topics)
        topics['qid'] = topics['qid'].astype(str)
        print(topics)
        results = pt.Experiment(
            [dr],
            topics,
            qrels,
            eval_metrics=[RR@10, R@10, R@100],
            filter_by_topics=True,
            dataframe=True, perquery = True
        )

        if not os.path.exists(f"./modified/{sys.argv[1]}"):
            os.makedirs(f"./modified/{sys.argv[1]}")

        results.to_csv(f"./modified/{sys.argv[1]}/{filename}_experiment_DENSE_perquery.csv")

        print(f"Saved results .csv in: ./modified/{sys.argv[1]}/{filename}_experiment_DENSE_perquery.csv")
         
else:
    results = pt.Experiment(
        [dr],
        topics,
        qrels,
        eval_metrics=[RR@10, R@10, R@100],
        filter_by_topics=True,
        dataframe=True, perquery = True
    )

    if not os.path.exists("./baseline/"):
        os.makedirs("./baseline/")

    results.to_csv(f"./baseline/{sys.argv[1]}_DENSE_perquery.csv")

    print(f"Saved results .csv in: ./baseline/{sys.argv[1]}_DENSE.csv")

# RR - Reciprocal Rank, absolute ordering
# nDCG - ordering again
# R - recall - important metric for retrieval - relevant docs retrieved/ all relevant 
# P - precision - relevant documents retrieved/ all retrieved
# SetP  - Set precision - relecant docs retrieved/ all relevant + all retrieved (Jaccard)
