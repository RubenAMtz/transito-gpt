import numpy as np
import openai
import pandas as pd
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from constants import COMPLETIONS_MODEL, EMBEDDING_MODEL, MAX_SECTION_LEN, SEPARATOR, ENCODING, encoding, separator_len, COMPLETIONS_API_PARAMS


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.texto) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
         "articulo", "0", "1", ... up to the length of the embedding vectors and "tokens" which is empty
    """
    df = pd.read_csv(fname, header=0)
    # remove index column
    df = df.drop(columns=["Unnamed: 0"])


    max_dim = max([int(c) for c in df.columns if c != "articulo" and c != "tokens"])
    return {
           (r.articulo): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def save_embeddings(fname: str, embeddings: dict[tuple[str], list[float]]):
    """
    Save the document embeddings and their keys to a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "articulo", "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.DataFrame(embeddings)
    df = df.T
    df = df.reset_index()
    df = df.rename(columns={"index": "articulo"})
    df.to_csv(fname, header=True)
    

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        # print(section_index)
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        # print(chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.texto.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Contesta la pregunta de la forma más honesta posible apoyandote del contexto proporcionado, haz referencias al contexto si es necesario, cuando hagas referencia al contexto di "Segun el documento oficial ", y si la respuesta no está contenida en el texto a continuación, diga "Una disculpa, no lo sé, intenta siendo más específico. Asegúra que la respuesta tenga que ver con la pregunta Q:".\n\nContexto:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def get_section_text(section_index: tuple[float, tuple[str, str, str]], df: pd.DataFrame) -> str:
    """
    Return the text of the section with the given index.
    """
    section_index = section_index[1]
    return df.loc[section_index].texto


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)


    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")