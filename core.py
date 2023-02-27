import numpy as np
import openai
import pandas as pd
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from constants import COMPLETIONS_MODEL, EMBEDDING_MODEL, MAX_SECTION_LEN, SEPARATOR_A, ENCODING, encoding, \
    COMPLETIONS_API_PARAMS, SUMMARIZATION_HEADER
from constants import HEADER, ENCODING_MODEL, QUESTIONS
import tiktoken
import re
from pprint import pprint
import json
import asyncio
import aiohttp


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


def load_dataset(fname: str, df_index_columns: list[str], store_path: str='dataset_store'):
    """
    Read the dataset from a CSV.

    Args:
        fname (str): The name of the CSV file.
        df_index_columns (list[str]): The names of the columns to use as the index.
        store_path (str, optional): The path to the directory where the CSV file is stored. Defaults to 'dataset_store'.

    Returns:
        pd.DataFrame: The dataset.
    """
    # load the csv file
    df = pd.read_csv(f'{store_path}/' + fname)
    # drop Unnamed: 0 column
    df = df.drop(columns=['Unnamed: 0'])
    df = df.set_index(df_index_columns)
    print(f"{len(df)} rows in the data.")
    df.sample(5)
    return df


def load_embedding(fname: str, index_columns: list[str], store_path='embeddings_store') -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
         index_columns[0], index_columns[1], ..., index_columns[n], "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.read_csv(f'{store_path}/' + fname, header=0)
    # remove index column
    df = df.drop(columns=["Unnamed: 0"])

    # get the maximum dimension of the embedding vectors, ignoring the index columns and the tokens column
    max_dim = max([int(c) for c in df.columns if c not in index_columns]) + 1
    return {
        (row.ley, row.articulo, row.parte): [row[str(i)] for i in range(max_dim)] for _, row in df.iterrows()
    }


def save_embeddings(fname: str, index_columns: list[str], embeddings: dict[tuple[str], list[float]], store_path='embeddings_store'):
    """
    Save the document embeddings and their keys to a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        index_columns[0], index_columns[1], ..., index_columns[n], "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.DataFrame(embeddings)
    df = df.transpose()
    df.columns = [str(i) for i in range(len(df.columns))]
    df = df.reset_index()
    df.columns = index_columns + [str(i) for i in range(len(df.columns) - len(index_columns))]

    df.to_csv(f'{store_path}/'+ fname)
    

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


def relevant_documents_to_text(relevant_documents: list[(float, (str, str))], context_df: pd.DataFrame) -> str:
    """
    Return the text of the most relevant documents, separated by a separator.
    """
    return SEPARATOR_A.join([
        context_df.loc[doc_index].texto for _, doc_index in relevant_documents
    ])


def clean_query(query: str) -> str:
    """
    Remove punctuation and other characters from the query.
    """
    return re.sub(r"[^\w\s]", "", query).lower()


def get_section_text(section_index: tuple[float, tuple[str, str, str]], df: pd.DataFrame) -> str:
    """
    Return the text of the section with the given index.
    """
    section_index = section_index[1]
    return df.loc[section_index].texto


def ask_gpt(
    prompt: str,
    show_prompt: bool = False
) -> str:
    """
    Ask GPT-3 to complete the prompt.
    """
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


async def ask_gpt_async(
    prompt: str,
    session: aiohttp.ClientSession,
    show_prompt: bool = False
) -> str:
    """
    Ask GPT-3 to complete the prompt.
    """
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


async def run_ask_gpt_async(
    prompts: list[str],
    show_prompt: bool = False
) -> list[str]:
    """
    Ask GPT-3 to complete the prompt.
    """
    async with aiohttp.ClientSession() as session:
        ret = await asyncio.gather(
            *[ask_gpt_async(prompt, session, show_prompt=show_prompt) for prompt in prompts]
        )
    return ret


def parse_answer(answer: str) -> dict:
    """
    Turns a GPT-3 answer into a dictionary.

    Args:
        answer (str): The answer to be cleaned.
    
    Returns:
        dict: The cleaned answer.
    """
    answer = eval("{" + "".join(answer.split("\n")[1:-1]).replace("'", '"') + "}")
    return answer
    