import pyterrier as pt
from pyterrier_dr import NumpyIndex, TctColBert
import torch
from pathlib import Path

pt.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = TctColBert('castorini/tct_colbert-v2-msmarco') 
model.model.to(device)

dataset = pt.get_dataset('irds:beir/arguana')

index_pipeline = model >> NumpyIndex('indices/arguana_tct_colbert_v2_msmarco.np')

index_pipeline.index(dataset.get_corpus_iter())


index_path = Path("/home/obez/InformationRetrieval/indices/arguana_bm25_index")
index_path.mkdir(parents=True, exist_ok=True)

indexer = pt.index.IterDictIndexer(
    str(index_path),
    meta={"docno": 100, "text": 131072},
)

index_ref = indexer.index(dataset.get_corpus_iter())