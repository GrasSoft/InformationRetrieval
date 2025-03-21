import pyterrier as pt
from pyterrier_dr import FlexIndex, TctColBert

model = TctColBert()

index = FlexIndex("arguana.flex")

retr_pipeline = model >> index.np_retriever()

print(retr_pipeline.search('African countries'))

# retr_pipeline = model >> index.faiss_hnsw_retriever()
