import sys
import json
import requests

# Default model
model = "llama3.2"
message = "Hello"


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

# API payload
data = {
    "model": model,
    "prompt": message,
    "stream": False,
    "options": {
        "seed": 42,
        "temperature": 0.5,
        "max_tokens": 150,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1
    }
}

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
	    return None
	    print(f"Error: {response.status_code}, {response.text}")
	    sys.exit(1)

infer(model, message)
