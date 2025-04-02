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


dataset = pt.get_dataset('irds:beir/arguana')


newpath = "./query_short/arguana/"

if not os.path.exists(newpath):
    os.makedirs(newpath)


prompt_zero_shot_expansion = """
	Perform query expansion on the given query. Enhance the query by adding additional, contextually relevant terms that could help retrieve documents lacking direct lexical overlap with the original query. 
 	Add synonyms or related terms to broaden its scope. Answer with only the query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""

prompt_few_shot_expansion = """
	Perform query expansion on the given query. Enhance the query by adding additional, contextually relevant terms that could help retrieve documents lacking direct lexical overlap with the original query.
	Add synonyms or related terms to broaden its scope. Here are a few examples:
 	Input: a deficiency of vitamin b12 increases blood levels of homocysteine 
  	Output: How does vitamin B12 deficiency or cobalamin deficiency lead to high homocysteine levels and increase the risk of heart disease or atherosclerosis. 
    Input: Stopping Heart Disease in Childhood 
    Output: Preventing heart disease in children, pediatric cardiovascular health, early intervention strategies 
    Answer with only the expanded query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""

prompt_zero_shot_reduction = """
	Perform query reduction on the given query. Your task is to condense the query to its core meaning while removing any extraneous words. Try to retain as much meaning as possible from the original query even if it comes at a cost of smaller reductions. If the query is already brief, return it unchanged. 
	Answer with only the query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""
   
prompt_few_shot_reduction = """
	Perform query reduction on the given query. Your task is to condense the query to its core meaning while removing any extraneous words. 
	Try to retain as much meaning as possible from the original query even if it comes at a cost of smaller reductions.
 	If the query is already brief, return it unchanged. Here are a few examples: 
  	Input: a deficiency of vitamin b12 increases blood levels of homocysteine 
    Output: Vitamin B12 deficiency increases homocysteine levels 
    Input: Stopping Heart Disease in Childhood. 
    Output: Preventing childhood heart disease 
    Input: how can i get a cork out of a wine bottle without a corkscrew 
    Output: How to remove a cork without a corkscrew. 
    Answer with only the query and nothing else, do not announce it, do not use search operators, do not use special characters, please."""

prompt_zero_shot_keywords = """
	Extract keywords from the given query. Identify and list the core terms that capture the essence of the query.
 	Answer with only the keywords as a list and nothing else, do not announce it, do not use search operators, do not use special characters, please."""

prompt_few_shot_keywords = """
	Extract keywords from the given query. Identify and list the core terms that capture the essence of the query. Return only the keywords as a list. Here are a few examples: 
 	Input: a deficiency of vitamin b12 increases blood levels of homocysteine 
  	Output: vitamin B12 deficiency, homocysteine 
    Input: Stopping Heart Disease in Childhood 
    Output: heart disease, childhood, prevention 
    Input: how can i get a cork out of a wine bottle without a corkscrew 
    Output: cork removal, wine bottle, alternative methods.
    Answer with only the keywords as a list and nothing else, do not announce it, do not use search operators, do not use special characters, please."""


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

prompts = [prompt_zero_shot_expansion, prompt_few_shot_expansion, prompt_zero_shot_reduction, prompt_few_shot_reduction, prompt_zero_shot_keywords, prompt_few_shot_keywords]
names = ["prompt_zero_shot_expansion", "prompt_few_shot_expansion", "prompt_zero_shot_reduction", "prompt_few_shot_reduction", "prompt_zero_shot_keywords", "prompt_few_shot_keywords"]



for idx, prompt in enumerate(prompts):
	rows = []
	for index, row in (dataset.get_topics()[:500]).iterrows():
		query = row["query"]

		# print(query)

		message = f"This is the query: {query}. {prompt}" 


		# API payload
		data = {
			"model": model,
			"prompt": message,
			"stream": False,
			"options": {
				"seed": 42,
				"temperature": 0.5,
				"max_tokens": 250, # change for arguana
				"presence_penalty": 0.1,
				"frequency_penalty": 0.1
			}
		}

		response = infer(model, message, False)

		if response:
			rows.append({
					"qid": row["qid"],
					"query": response["response"],
					"orig_query": query
				})




	df_topics = pd.DataFrame(rows, columns=["qid", "query", "orig_query"])
	df_topics.to_csv(f"./query_short/arguana/{names[idx]}.csv", index=False)


