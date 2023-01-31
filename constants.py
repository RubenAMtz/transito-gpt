import openai
import tiktoken
import os

HEADER = """Las preguntas tratarán de describir una situación, y quizá vengan acompañadas de una pregunta. Contesta el comentario y/o pregunta apoyándote del contexto proporcionado y de tu conocimiento general en el tema. Describe tu respuesta con un tono amigable y formal, trata de ponerte en los zapatos del usuario. Se claro y trata de evitar palabras ambiguas. Si la pregunta o comentario no tienen que ver con leyes de tránsito no respondas, repito, no contestes a preguntas o comentarios que no sean relacionadas a las Leyes de Tránsito.\n\nContexto:\n"""

# HEADER = """Contesta la pregunta de la forma más honesta posible apoyandote del contexto proporcionado, haz referencias al contexto si es necesario, cuando hagas referencia al contexto di algo como "Segun el documento oficial ", y si la respuesta no está contenida en el texto a continuación, diga "Una disculpa, no lo sé, intenta siendo más específico. Asegúrate que la respuesta tenga que ver con la pregunta Q y describe tu respuesta con un tono amigable y formal. Enriquice la respuesta con datos relevantes".\n\nContexto:\n

# Para informacion más clara y precisa, te invitamos a leer los articulos en los que está basada esta respuesta

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
ENCODING_MODEL = 'p50k_base' # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# try to load api key from file, otherwise use environment variable but do not raise error if not found
if os.path.exists("keys.txt"):
    openai.api_key_path = "keys.txt"
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}