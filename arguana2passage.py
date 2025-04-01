import json
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import pyterrier as pt


def query_iterator_from_df(df):
    """Convert queries DataFrame to BEIR-compatible iterator"""
    for _, row in df.iterrows():
        yield {
            'qid': str(row['qid']),      # Required field
            'query': str(row['query']),   # Required field
            # Add any additional metadata if needed
            'metadata': {}
        }

# 3. Qrels Iterator (for qrels.tsv)
def qrel_iterator_from_df(df):
    """Convert qrels DataFrame to BEIR-compatible iterator"""
    for _, row in df.iterrows():
        yield {
            'qid': str(row['qid']),      # Required field
            'docno': str(row['docno']),   # Required field
            'label': int(row['label'])    # Required field (relevance score)
        }

def download_arguana_dataset():
    """Download Arguana dataset using PyTerrier"""
    if not pt.started():
        pt.init()
    
    print("Downloading Arguana dataset...")
    dataset = pt.get_dataset("irds:beir/arguana")
    docs = dataset.get_corpus_iter()
    queries = query_iterator_from_df(dataset.get_topics())
    qrels = qrel_iterator_from_df(dataset.get_qrels())
    
    return docs, queries, qrels

def split_into_passages(text, passage_length=10, stride=8, min_passage_length=50):
    """
    Split document into passages with specified length and stride.
    
    Args:
        text: Input document text
        passage_length: Number of sentences per passage
        stride: Number of sentences to move forward for next passage
        min_passage_length: Minimum character length for a passage to be included
    
    Returns:
        List of passage texts with their sentence ranges
    """
    sentences = sent_tokenize(text)
    passages = []
    
    # If document has fewer sentences than passage length, return the whole document
    if len(sentences) <= passage_length:
        passage = ' '.join(sentences)
        if len(passage) >= min_passage_length:
            return [(passage, (1, len(sentences)))]
        return []
    
    # Create passages with overlap
    for i in range(0, len(sentences) - passage_length + 1, stride):
        passage_text = ' '.join(sentences[i:i+passage_length])
        if len(passage_text) >= min_passage_length:
            passages.append((passage_text, (i+1, i+passage_length)))
    
    # Add remaining sentences if any
    remaining_start = len(sentences) - passage_length
    if remaining_start % stride != 0:
        passage_text = ' '.join(sentences[remaining_start:])
        if len(passage_text) >= min_passage_length:
            passages.append((passage_text, (remaining_start+1, len(sentences))))
    
    return passages

def create_beir_dataset(docs, queries, qrels, output_dir):
    """
    Create BEIR-compatible passage dataset from PyTerrier data
    
    Args:
        docs: PyTerrier document iterator
        queries: PyTerrier queries iterator
        qrels: PyTerrier qrels iterator
        output_dir: Directory to save BEIR-compatible dataset
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    corpus_dir = os.path.join(output_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    
    # Process corpus file
    output_corpus_file = os.path.join(corpus_dir, "corpus.jsonl")
    docid_to_passageids = {}
    
    with open(output_corpus_file, "w", encoding="utf-8") as fout:
        passage_id_counter = 1
        
        for doc in tqdm(docs, desc="Processing documents"):
            doc_id = str(doc["docno"])
            text = doc["text"]
            title = doc.get("title", "")
            
            # Split document into passages (10 sentences with stride of 2)
            passages = split_into_passages(text, passage_length=10, stride=8)
            
            passage_ids = []
            for i, (passage_text, (start_sent, end_sent)) in enumerate(passages):
                passage_id = f"{doc_id}_p{passage_id_counter}"
                passage_ids.append(passage_id)
                
                # Create BEIR-compatible passage entry
                passage_entry = {
                    "_id": passage_id,
                    "title": title,
                    "text": passage_text,
                    "metadata": {
                        "document_id": doc_id,
                        "passage_number": i+1,
                        "passage_start_sentence": start_sent,
                        "passage_end_sentence": end_sent,
                        "original_document_title": title
                    }
                }
                
                fout.write(json.dumps(passage_entry) + "\n")
                passage_id_counter += 1
            
            docid_to_passageids[doc_id] = passage_ids
    
    # Process queries
    output_queries_file = os.path.join(output_dir, "queries.jsonl")
    with open(output_queries_file, "w", encoding="utf-8") as fout:
        for query in tqdm(queries, desc="Processing queries"):
            query_entry = {
                "_id": str(query["qid"]),
                "text": query["query"],
                "metadata": {}
            }
            fout.write(json.dumps(query_entry) + "\n")
    
    # Process qrels (map document IDs to passage IDs)
    output_qrels_dir = os.path.join(output_dir, "qrels")
    os.makedirs(output_qrels_dir, exist_ok=True)
    
    # Group qrels by query ID
    qrels_dict = {}
    for qrel in qrels:
        query_id = str(qrel["qid"])
        doc_id = str(qrel["docno"])
        relevance = qrel["label"]
        
        if query_id not in qrels_dict:
            qrels_dict[query_id] = []
        qrels_dict[query_id].append((doc_id, relevance))
    
    # Write qrels files
    for split in ["train", "dev", "test"]:
        output_qrel_path = os.path.join(output_qrels_dir, f"{split}.tsv")
        with open(output_qrel_path, "w", encoding="utf-8") as fout:
            for query_id, doc_relevance_pairs in qrels_dict.items():
                for doc_id, relevance in doc_relevance_pairs:
                    if doc_id in docid_to_passageids:
                        for passage_id in docid_to_passageids[doc_id]:
                            fout.write(f"{query_id}\t0\t{passage_id}\t{relevance}\n")
    
    print(f"Successfully created BEIR-compatible passage dataset in {output_dir}")

def main():
    # Download dataset
    docs, queries, qrels = download_arguana_dataset()
    
    # Create BEIR-compatible passage dataset
    output_directory = "arguana-passage-beir"
    create_beir_dataset(docs, queries, qrels, output_directory)

if __name__ == "__main__":
    # Download NLTK punkt tokenizer if not already available
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()