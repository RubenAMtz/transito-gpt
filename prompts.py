import numpy as np
import pandas as pd
import tiktoken
from constants import ENCODING_MODEL, MAX_SECTION_LEN, HEADER, SEPARATOR_A,\
    SEPARATOR_B
from core import order_document_sections_by_query_similarity, clean_query, \
    get_embedding, compute_doc_embeddings, load_dataset, load_embedding, vector_similarity
from pprint import pprint
from typing import Tuple, Any, Union



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
        
        chosen_sections_len += document_section.tokens + len(gpt_tokenizer.encode(SEPARATOR_A))
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


def construct_append_prompt(prompt: str, query: str,) -> str:
    """
    Constructs a prompt for a task specified by
    the prompt parameter. It appends the query
    to the prompt. 
    """
    query = clean_query(query)
    return prompt + "\n" + query + "\n"


# def construct_prompt_form(prompt: str, **kwargs) -> str:
    



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
        
        chosen_sections_len += document_section.tokens + len(gpt_tokenizer.encode(SEPARATOR_A))
        # print(chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.texto.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    return SUMMARIZATION_HEADER + "".join(chosen_sections)


def construct_form_prompt(question: str, context_embeddings: dict[(str, str), np.array], context_df: pd.DataFrame, prompt: str) -> str:
    """
    Constructs a prompt for a task specified by
    the prompt parameter. It fills out the 'blanks'
    in the prompt with the keyword arguments.
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    # used to calculate the length of the header's prompt encoding
    gpt_tokenizer = tiktoken.get_encoding(ENCODING_MODEL)
    chosen_sections_len = len(gpt_tokenizer.encode(prompt))

    chosen_sections = []
    chosen_sections_indexes = []
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = context_df.loc[section_index]
        # get the value of the tokens column
        
        chosen_sections_len += document_section.tokens + len(gpt_tokenizer.encode(SEPARATOR_A))
        # print(chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        # replace the prompt with the section's text

            
        chosen_sections.append(SEPARATOR.join([
            document_section.articulo,
            document_section.parte,
            document_section.texto
        ]))
        chosen_sections_indexes.append(section_index)
        
    prompt = HEADER + SEPARATOR.join(chosen_sections)
    return prompt, chosen_sections_indexes


class PromptBuilder:

    def __init__(self, prompt_template: str) -> None:
        self.prompt = prompt_template
        self.relevant_documents = {}
        self.chosen_documents = []
        self.prompt_built = None

    # we will define a max_section_length manually, and then we will add the sections until we reach that length
    def _section_length_controller(self, documents_section: Union[str,list[str]], max_section_length: int, separator: str) -> str:
        
        # used to calculate the length of the header's prompt encoding
        gpt_tokenizer = tiktoken.get_encoding(ENCODING_MODEL)
        chosen_sections_len = len(gpt_tokenizer.encode(self.prompt))

        chosen_sections = []

        if isinstance(documents_section, str):
            chosen_sections_len = len(gpt_tokenizer.encode(documents_section)) + len(gpt_tokenizer.encode(separator))
            if chosen_sections_len > max_section_length:
                raise ValueError("The section is too long")
            chosen_sections.append(separator + documents_section.replace("\n", " "))

        elif isinstance(documents_section, list):
            # Add contexts until we run out of space.
            # maybe a bit of a redundant use of variables, TODO: refactor if needed
            for (document, relevant_document) in zip(documents_section, self.relevant_documents):
                chosen_sections_len += len(gpt_tokenizer.encode(document)) + len(gpt_tokenizer.encode(separator))
                if chosen_sections_len > max_section_length:
                    break

                chosen_sections.append(separator + document.replace("\n", " "))
                # save the selected documents
                self.chosen_documents.append((relevant_document, self.relevant_documents[relevant_document]))
            
        return "".join(chosen_sections)


    def get_relevant_documents(self, query: str, context_dict: dict[(str, str), np.array], dataset: pd.DataFrame, top_n=20) -> dict[Tuple[str, str], str]:
        """
        Orchestrates the process of getting the most relevant documents
        for a given query and returns the index and a list strings 
        corresponding to the text of the most relevant sections. It 
        arbitrarily chooses the first 10 sections.

        Args:
            query (str): The query to be used.  
            context_dict (dict[(str, str), np.array]): A dictionary of the 
                form {(document_id, section_id): embedding}.
            dataset (pd.DataFrame): A dataframe containing the text of the
                documents.
            top_n (int, optional): The number of documents to return. Defaults to 20.
        Returns:
            tuple: A dictionary with keys in the form of index tuples and the 
            text of the most relevant sections as value.
        """
        query_embedding = get_embedding(query)
    
        relevant_documents = sorted([
            (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in context_dict.items()
        ], reverse=True)[:top_n]

        
        relevant_docs = dict()
        for _, doc_index in relevant_documents:
            relevant_docs[doc_index] = dataset.loc[doc_index].texto
            # self.relevant_documents_indexes.append(doc_index)
        
        self.relevant_documents.update(relevant_docs)
        return relevant_docs


    def build_prompt(self, sections: dict[str,dict[str, int]]) -> str:
        """
        Constructs a prompt for a task specified by
        the prompt parameter. It fills out the 'blanks'
        in the prompt with the text_sections keywords.

        Args:
            sections (dict[str,dict[str, int]]): A dictionary with the sections
                to be added to the prompt. The keys are the names of the sections,
                the values are dictionaries with the keys 'text', 'max_length' and
                'separator'. The 'text' key is the text to be added to the prompt,
                the 'max_length' key is the maximum token length of the section and 
                the 'separator' key is the string separator to be used.
                For example:
                {
                    'section_01': {
                        'text': 'This is the header',
                        'max_length': 100,
                        'separator': '- '
                    },
                    'section_02': {
                        'text': ['This is the first doc', 'This is the second doc'],
                        'max_length': 1024,
                        'separator': '\n* '
                    }
                }
        Returns:
            str: The prompt with the sections added.
        """
        assert self.prompt is not None, "You must set the prompt template first"
        # get the subset of text from text_sections using the sections_lengths
        # and then build the prompt
        prompt = self.prompt
        for section in sections:
            section_text = sections[section]['text']
            section_max_length = sections[section]['max_length']
            section_separator = sections[section]['separator']
            prompt = prompt.replace(f"{{{section}}}", self._section_length_controller(section_text, section_max_length, section_separator))
        
        self.prompt_built = prompt
        return prompt


    def validate_prompt_length(self) -> None:
        """
        Validates that the prompt is not longer than the MAX_SECTION_LEN.
        """
        gpt_tokenizer = tiktoken.get_encoding(ENCODING_MODEL)
        assert MAX_SECTION_LEN >= len(gpt_tokenizer.encode(self.prompt_built)), f"The prompt is too long {len(gpt_tokenizer.encode(self.prompt_built))} > {MAX_SECTION_LEN}"
        print("The prompt is valid: ", len(gpt_tokenizer.encode(self.prompt_built)), " tokens")

        
        

    

# mock-up:
# prompt_builder = PromptBuilder()

#  prompt_builder.get_relevant_documents(query, context_df) # returns both index and text of the most relevant sections: (index, text)

# prompt_builder.build_prompt(prompt, {'section_01': {'text':'some text', 'max_length': 512}, 'section_02': {'text':'some other text', 'max_length': 128}})
# or
# prompt_builder.build_prompt(prompt, {'section_01': {'text': ['some text', 'some more text'], 'max_length': 512}, 'section_02': {'text':'some other text', 'max_length': 128}

# prompt_builder.get_texts_indexes() # returns the index of the text sections used in the prompt

        # for key, value in text_sections.items():
        #     prompt = prompt.replace(f"{{{key}}}", value)
        # return prompt
