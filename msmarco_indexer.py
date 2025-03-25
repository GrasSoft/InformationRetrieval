import pyterrier as pt
from pyterrier_dr import NumpyIndex, TctColBert
import torch
from pathlib import Path
import ir_datasets
import pandas as pd

pt.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco') 
model.model.to(device)

dataset = pt.get_dataset('irds:msmarco-passage')

index_pipeline = model >> NumpyIndex('indices/msmarco_passages_tct_colbert_v2_hnp_msmarco.np')

index_pipeline.index(dataset.get_corpus_iter())


# index_path = Path("/home/obez/InformationRetrieval/indices/msmarco_passages_bm25_index")
# index_path.mkdir(parents=True, exist_ok=True)

# indexer = pt.index.IterDictIndexer(
#     str(index_path),
#     meta={"docno": 200, "text": 531072},
# )

# index_ref = indexer.index(dataset.get_corpus_iter())
