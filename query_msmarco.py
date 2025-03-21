import pyterrier as pt

# Start PyTerrier
if not pt.started():
    pt.init()

from pyterrier_dr import FlexIndex

# Load known dense FlexIndex for MS MARCO Document (tasb variant)
index = FlexIndex.from_dataset(
    dataset="msmarco_document",
    variant="tasb-distilroberta-base-msmarco"
)

# Create a retriever from the index
retriever = index.query()

# Run a test query
query = "What is the capital of France?"
results = retriever.search(query)

# Show top results
print(results[['docno', 'score']].head())

