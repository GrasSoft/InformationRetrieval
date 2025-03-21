# HOW TO USE THE SCRIPTS IN HERE
## prompt.py
- `prompt.py` - is a python file that handles prompting the LLM. To run it by itself just type `python prompt.py` and it will run with default parameters (LLaMa model and the message "Hello"). 
- If you want to change the model provide it with a parameter `python prompt.py llama` (the only options are llama fro llama3.2 and gemma for gemma3:12b) gemma is larger thus slower out of the 2
- If you want to change the message provide it as a parameter, can be first or second `python prompt.py llama "Message"` or `python prompt.py "Message"`
- If you want to use it in another python file, simply import it and call the function `infer` with the model and message parameter - Note use Print flag to stop it from printing to the console,
 the function return a python object with the model used, date and response. In case something fails it will return None and print error. Provide the model with the full name in this parameter as per this table https://github.com/ollama/ollama/blob/main/README.md#quickstart
