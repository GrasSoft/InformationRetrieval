import sys
import json
import requests
import pyterrier as pt
import os
import pandas as pd

# Default model
model = "llama3.2"

# Check command-line arguments
if len(sys.argv) > 1:
	if sys.argv[1] == "llama":
		model = "llama3.2"
	elif sys.argv[1] == "gamma":
		model = "gemma3:12b"
	else:
		message = sys.argv[1]  # If the first argument is not 'llama' or 'gamma', treat it as the message

# Check if a second argument is provided for the message
if len(sys.argv) > 2:
	message = sys.argv[2]


dataset = pt.get_dataset('irds:beir/trec-covid')


newpath = "./query_short/trec-covid-query2doc/"

if not os.path.exists(newpath):
    os.makedirs(newpath)

index_bm25 = pt.IndexFactory.of(f"/home/obez/InformationRetrieval/indices/trec_covid_bm25_index")


bm25 = pt.terrier.Retriever(
        index_bm25,
        wmodel="BM25",
        metadata=["docno", "text"],
        properties={"termpipelines": "Stopwords,PorterStemmer"},
        controls={"qe": "off"},
    )




def infer(model="llama3.2", message="Hello", Print=True):
	# API endpoint
	url = "http://localhost:11434/api/generate"

	# Send request
	response = requests.post(url, json=data)

	# Process response
	if response.status_code == 200:
		result = response.json()
		formatted_result = {
		"model": result.get("model"),
		"created_at": result.get("created_at"),
		"response": result.get("response")
		}
		if Print is True:
			print(json.dumps(formatted_result, indent=2))
		return formatted_result
	else:
		print(f"Error: {response.status_code}, {response.text}")
		return None

new_queries = []

for index, row in (dataset.get_topics()[:500]).iterrows():

    results = (bm25 % 3).search(row["query"])

    if len(results) == 3:
        prompt = f"""
            Write a list of keywords for the given query based on the context:
            Context: 
            {results["text"][0]} 
            {results["text"][1]} 
            {results["text"][2]} 
            Query: {row["query"]} 
            Keywords:
        """
    
        # API payload
        data = {
			"model": model,
			"prompt": prompt,
			"stream": False,
			"options": {
				"seed": 42,
				"temperature": 0.5,
				"max_tokens": 300,
				"presence_penalty": 0.1,
				"frequency_penalty": 0.1
			}
		}

        response = infer(model, data, False)

        if response:
            new_queries.append({
                    "qid": row["qid"],
                    "query": response["response"],
                    "orig_query": row["query"]
                })


df_topics = pd.DataFrame(new_queries, columns=["qid", "query", "orig_query"])
df_topics.to_csv(f"./query_short/trec-covid-query2doc/trec-covid-query2doc.csv", index=False)


