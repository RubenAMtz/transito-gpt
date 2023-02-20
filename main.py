from core import load_embedding, load_dataset, \
    order_document_sections_by_query_similarity, \
    ask_gpt, parse_answer, relevant_documents_to_text
import pandas as pd
import argparse

from prompts_store.classification.intent import INTENT_PROMPT
from prompts_store.entity_recognition.query_entities import QUERY_ENTITY_RECOGNITION_PROMPT
from prompts_store.ask_gpt.user_action_vs_law import USER_ACTION_VS_LAW_PROMPT
from prompts_store.ask_gpt.officer_action_vs_law import OFFICER_ACTION_VS_LAW_PROMPT
from prompts_store.ask_gpt.user_and_officer_vs_law import USER_AND_OFFICER_ACTION_VS_LAW_PROMPT

from prompts import PromptBuilder
from constants import SEPARATOR_A, SEPARATOR_A_, SEPARATOR_B, SEPARATOR_B_

argsparser = argparse.ArgumentParser()
# argument is list of strings
argsparser.add_argument('--query', '-q', nargs='+', required=True)
args = argsparser.parse_args()
query = ' '.join(args.query)

# our query is submitted to a NER model, and we get the following entities:
# query = 'me puedo estacionar en doble fila?'
# query = 'me pueden multar dentro de mi casa?'
# query = 'me pueden multar por no usar el cinturon?'
query = 'me cruce un semaforo y me llevaron al juzgado, tiene esa facultad un estatal?'

# create a prompt and ask to run entity recognition
ner_prompt_builder = PromptBuilder(QUERY_ENTITY_RECOGNITION_PROMPT)
ner_prompt = ner_prompt_builder.build_prompt({'section_01': {'text': query, 'max_length': 512, 'separator': SEPARATOR_B}})
ner = parse_answer(ask_gpt(ner_prompt))

# create a prompt to classify the intent of the question
intent_prompt_builder = PromptBuilder(INTENT_PROMPT)
intent_prompt = intent_prompt_builder.build_prompt({'section_01': {'text': query, 'max_length': 512, 'separator': SEPARATOR_B}})
intent = parse_answer(ask_gpt(intent_prompt))


# we read two different datasets and embeddings
# dataset_names = ['leydetransitov4.csv', 'leyseguridadpublicav1.csv']
# df_index_columns = ['articulo', 'parte']
dataset_names = ['leydetransitov4 - Copy.csv', 'leyseguridadpublicav1 - Copy.csv']
df_index_columns = ['ley','articulo', 'parte']

# embeddings_names = ['embeddingsv4.csv', 'embeddings_seguridad_publica_v1.csv']
# embedding_index_columns = ['articulo', 'parte']
embeddings_names = ['embeddings_transito_v4 - Copy.csv', 'embeddings_seguridad_publica_v1 - Copy.csv']
embedding_index_columns = ['ley','articulo', 'parte']

leydetransito_df = load_dataset(dataset_names[0], df_index_columns, store_path='./datasets_store')
leyseguridadpublica_df = load_dataset(dataset_names[1], df_index_columns, store_path='./datasets_store')

leydetransito_embeddings = load_embedding(embeddings_names[0], embedding_index_columns, store_path='./embeddings_store')
leyseguridadpublica_embeddings = load_embedding(embeddings_names[1], embedding_index_columns, store_path='./embeddings_store')

# run a logic state depending on the question intent and entities
if intent['user_action_vs_law'] and not intent['officer_action_vs_law']:
    if ner['user_action']:
        # pick the prompt template
        user_action_vs_law_prompt_builder = PromptBuilder(USER_ACTION_VS_LAW_PROMPT)
        # we get the most relevant sections for the user action
        LDT_relevant_docs_user_action = user_action_vs_law_prompt_builder.get_relevant_documents(ner['user_action'], leydetransito_embeddings, leydetransito_df)
        # fill out the prompt 'form'
        user_action_vs_law_prompt = user_action_vs_law_prompt_builder.build_prompt({
                'user_action': {
                    'text': ner['user_action'],
                    'max_length': 256,
                    'separator': SEPARATOR_B_
                },
                'LDT_relevant_docs_user_action': {
                    'text': list(LDT_relevant_docs_user_action.values()), 
                    'max_length': 1280, 
                    'separator': SEPARATOR_B
                }
            })
        # ask gpt to generate the answer
        user_action_vs_law_res = parse_answer(ask_gpt(user_action_vs_law_prompt))

elif intent['officer_action_vs_law'] and not intent['user_action_vs_law']:
    pass

elif intent['officer_action_vs_law'] and intent['user_action_vs_law']:
    if ner['officer_action_or_consequence']:
        # we get the most relevant sections, officer's actions have to be checked in both datasets
        user_and_officer_vs_law_prompt_builder = PromptBuilder(USER_AND_OFFICER_ACTION_VS_LAW_PROMPT)
        LDT_relevant_docs_officer_action = user_and_officer_vs_law_prompt_builder.get_relevant_documents(
            ner['officer_action_or_consequence'], 
            leydetransito_embeddings, 
            leydetransito_df
        )
        LSP_relevant_doc_officer_action = user_and_officer_vs_law_prompt_builder.get_relevant_documents(
            ner['officer_action_or_consequence'],
            leyseguridadpublica_embeddings,
            leyseguridadpublica_df
        )
        # user's action is checked only in the traffic law dataset
        LDT_relevant_docs_user_action = user_and_officer_vs_law_prompt_builder.get_relevant_documents(
            ner['user_action'],
            leydetransito_embeddings,
            leydetransito_df
        )

        # fill out the prompt 'form'
        user_and_officer_action_vs_law_prompt = user_and_officer_vs_law_prompt_builder.build_prompt({
                'user_action': {
                    'text': ner['user_action'],
                    'max_length': 256,
                    'separator': SEPARATOR_B_
                },
                'LDT_relevant_docs_user_action': {
                    'text': list(LDT_relevant_docs_user_action.values()), 
                    'max_length': 650, 
                    'separator': SEPARATOR_B
                },
                'officer_action_or_consequence': {
                    'text': ner['officer_action_or_consequence'],
                    'max_length': 256,
                    'separator': SEPARATOR_B_
                },
                'LDT_relevant_docs_officer_action': {
                    'text': list(LDT_relevant_docs_officer_action.values()),
                    'max_length': 650,
                    'separator': SEPARATOR_B
                },
                'LSP_relevant_doc_officer_action': {
                    'text': list(LSP_relevant_doc_officer_action.values()),
                    'max_length': 600,
                    'separator': SEPARATOR_B
                }
            })

        # ask gpt to generate the answer
        user_and_officer_action_vs_law_res = ask_gpt(user_and_officer_action_vs_law_prompt)
        print(user_and_officer_action_vs_law_res)
        # TODO: correctly track the chosen_docs
        
# elif question_class['complaint']:
#     # do something
#     pass





############################################################################################################
# we concatenate the two datasets and embeddings
# df = pd.concat([leydetransito_df, leyseguridadpublica_df])
# document_embeddings = pd.concat([leydetransito_embeddings, leyseguridadpublica_embeddings])

# summarized_documents_leydetransito = summarize_context(query, leydetransito_embeddings, leydetransito_df, show_prompt=True)
# summarized_documents_seguridadpublic = summarize_context(summarized_documents_leydetransito, leyseguridadpublica_embeddings, leyseguridadpublica_df, show_prompt=True)

# we get the different entities through entity recognition:
# mock-up:

# prompt = construct_prompt(query, leyseguridadpublica_embeddings, leyseguridadpublica_df)

# answer = answer_query_with_context(query, leyseguridadpublica_df, leyseguridadpublica_embeddings, show_prompt=True)


# print(f"\nP: {query}\nR: {answer}.\n\nPara informacion más clara y precisa, te invitamos a leer los articulos en los que está basada esta respuesta:")