import sys
import json
import requests
import pyterrier as pt
import os
import pandas as pd

pt.init()

# Default model
model = "llama3.2"

# Command-line model selection
if len(sys.argv) > 1:
	if sys.argv[1] == "llama":
		model = "llama3.2"
	elif sys.argv[1] == "gamma":
		model = "gemma3:12b"

# Dataset (for consistency)
dataset = pt.get_dataset('irds:beir/trec-covid')

# Input: CSV with qid, query, modified_query
input_file_path = "./query_misspelled_datasets/irds:beirtrec-covid.csv"  # adjust if needed
input_df = pd.read_csv(input_file_path)
input_df['qid'] = input_df['qid'].astype(str)

# Output directory
output_dir = "./query_short/trec-covid-misspelled/"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Zero-shot prompt variants
prompts = [
	{
		"name": "prompt_zero_shot_expansion",
		"text": """Perform query expansion on the given query. Enhance the query by adding additional, contextually relevant terms that could help retrieve documents lacking direct lexical overlap with the original query. 
Add synonyms or related terms to broaden its scope. Answer with only the query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""
	},
	{
		"name": "prompt_zero_shot_reduction",
		"text": """Perform query reduction on the given query. Your task is to condense the query to its core meaning while removing any extraneous words. Try to retain as much meaning as possible from the original query even if it comes at a cost of smaller reductions. If the query is already brief, return it unchanged. 
Answer with only the query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""
	},
	{
		"name": "prompt_zero_shot_keywords",
		"text": """Extract keywords from the given query. Identify and list the core terms that capture the essence of the query.
Answer with only the keywords as a list and nothing else, do not announce it, do not use search operators, do not use special characters, please."""
	}
]

# Inference function
def infer(model="llama3.2", message="Hello", Print=True):
	url = "http://localhost:11434/api/generate"
	response = requests.post(url, json=data)
	if response.status_code == 200:
		result = response.json()
		if Print:
			print(json.dumps(result, indent=2))
		return {
			"model": result.get("model"),
			"created_at": result.get("created_at"),
			"response": result.get("response")
		}
	else:
		print(f"Error: {response.status_code}, {response.text}")
		return None

# Apply each prompt to all modified queries
for prompt in prompts:
	rows = []
	for _, row in input_df.iterrows():
		orig_query = row["query"]
		query = row["modified_query"]

		message = f"This is the query: {query}. {prompt['text']}"

		data = {
			"model": model,
			"prompt": message,
			"stream": False,
			"options": {
				"seed": 42,
				"temperature": 0.5,
				"max_tokens": 250,
				"presence_penalty": 0.1,
				"frequency_penalty": 0.1
			}
		}

		response = infer(model, message, False)

		if response:
			rows.append({
				"qid": row["qid"],
				"query": response["response"],
				"orig_query": orig_query
			})

	df_topics = pd.DataFrame(rows, columns=["qid", "query", "orig_query"])
	output_file = os.path.join(output_dir, f"{prompt['name']}.csv")
	df_topics.to_csv(output_file, index=False)
	print(f"Saved: {output_file}")

