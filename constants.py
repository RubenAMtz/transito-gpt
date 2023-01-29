import openai
import tiktoken
import os

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 3500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# try to load api key from file, otherwise use environment variable but do not raise error if not found
if os.path.exists("keys.txt"):
    openai.api_key_path = "keys.txt"
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.1,
    "max_tokens": 250,
    "model": COMPLETIONS_MODEL,
}