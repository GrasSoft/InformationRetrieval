from  pyterrier_dr import FlexIndex,  RetroMAE, TctColBert
import torch
import pyterrier as pt

pt.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = TctColBert() 
model.model.to(device)

dataset = pt.get_dataset('irds:beir/arguana')
corpus_iter = dataset.get_corpus_iter()

index = FlexIndex("arguana.flex")

pipeline = model >> index.indexer()

batch_size = 100_000
batch = []
counter = 0


for i, doc in enumerate(corpus_iter):
    batch.append(doc)
    if len(batch) >= batch_size:
        print(f"Indexing batch {counter + 1} (docs {i - batch_size + 1} to {i})")
        pipeline.index(batch)
        batch = []
        counter += 1

# Final remaining batch
if batch:
    print(f"Indexing final batch {counter + 1}")
    pipeline.index(batch)

print(counter)
