import numpy as np
import openai
import pandas as pd
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from constants import COMPLETIONS_MODEL, EMBEDDING_MODEL, MAX_SECTION_LEN, SEPARATOR, ENCODING, encoding, \
    separator_len, COMPLETIONS_API_PARAMS, SUMMARIZATION_HEADER
from constants import HEADER, ENCODING_MODEL, QUESTIONS
import tiktoken
import re
from pprint import pprint
import json


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    query_embedding_length = len(result["data"][0]["embedding"])
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
        (row.articulo, row.parte): [row[str(i)] for i in range(max_dim)] for _, row in df.iterrows()
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


def contruct_prompts(question: str, contexts_embeddings: list[dict], dfs: list[pd.DataFrame]) -> str:
    """
    Fetch relevant context using the question and a set of pre-calculated documents embeddings.
    Return the prompt to be used for the GPT-3 completion.
    The prompt is constructed as follows:
    1. The question is used to find the most relevant document sections.
    2. The most relevant document sections are used to construct the prompt.

    Args:
        question (str): The question to be answered.
        contexts_embeddings (list[dict]): The pre-calculated document embeddings.
        dfs (list[pd.DataFrame]): The datasets.

    Returns:
        str: The prompt to be used for the GPT-3 completion.
    """
    pass


def construct_prompt_for_summarization(question, context_embeddings, context_df):
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    # used to calculate the length of the header's prompt encoding
    gpt_tokenizer = tiktoken.get_encoding(ENCODING_MODEL)
    chosen_sections_len = len(gpt_tokenizer.encode(SUMMARIZATION_HEADER))

    chosen_sections = []
    chosen_sections_indexes = []
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = context_df.loc[section_index]
        # get the value of the tokens column
        
        chosen_sections_len += document_section.tokens + separator_len
        # print(chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.texto.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    return SUMMARIZATION_HEADER + "".join(chosen_sections)


def construct_prompt(question: str, context_embeddings: dict, context_df: pd.DataFrame) -> str:
    """
    Fetch relevant context using the question and the pre-calculated document embeddings.
    Return the prompt to be used for the GPT-3 completion.
    The prompt is constructed as follows:
    1. The question is used to find the most relevant document sections.
    2. The most relevant document sections are used to construct the prompt.
    
    Args:
        question (str): The question to be answered.
        context_embeddings (dict): The pre-calculated document embeddings.
        context_df (pd.DataFrame): The dataset in context.

    Returns:
        str: The prompt to be used for the GPT-3 completion.
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    print(f"Most relevant document sections:")
    pprint(most_relevant_document_sections[:3])
    # used to calculate the length of the header's prompt encoding
    gpt_tokenizer = tiktoken.get_encoding(ENCODING_MODEL)

    chosen_sections = []
    chosen_sections_len = len(gpt_tokenizer.encode(HEADER))
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = context_df.loc[section_index]
        # get the value of the tokens column
        
        chosen_sections_len += document_section.tokens + separator_len
        # print(chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.texto.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    question = clean_query(question)
    question = "Resolvamos esto paso a paso para asegurarnos de que tenemos la respuesta correcta. Segun la ley y los articulos seleccionados " + question + " ?"
    
    return HEADER + "".join(chosen_sections) + "\n\n Pregunta: " + question + "\n Respuesta:"


def construct_prompt_generic(prompt: str, query: str,) -> str:
    """
    Constructs a prompt for a task specified by
    the prompt parameter. It appends the query
    to the prompt. 
    """
    query = clean_query(query)
    return prompt + "\n" + query + "\n"


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


def summarize_context(
    query: str,
    document_embeddings: dict[(str), np.array],
    df: pd.DataFrame,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt_for_summarization(
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

def clean_answer(answer: str) -> dict:
    """
    Turns a GPT-3 answer into a dictionary.

    Args:
        answer (str): The answer to be cleaned.
    
    Returns:
        dict: The cleaned answer.
    """
    answer = eval("{" + "".join(answer.split("\n")[1:-1]).replace("'", '"') + "}")
    return answer
    