from core import load_embedding, load_dataset, \
    order_document_sections_by_query_similarity, \
    ask_gpt, parse_answer, relevant_documents_to_text, \
    run_ask_gpt_async
import pandas as pd
import asyncio
import streamlit as st
import core
import logging
import sys
import os
from pdf import PDF
import datetime

from prompts_store.classification.intent import INTENT_PROMPT
from prompts_store.entity_recognition.query_entities import QUERY_ENTITY_RECOGNITION_PROMPT
from prompts_store.ask_gpt.user_action_vs_law import USER_ACTION_VS_LAW_PROMPT
from prompts_store.ask_gpt.officer_action_vs_law import OFFICER_ACTION_VS_LAW_PROMPT
from prompts_store.ask_gpt.user_and_officer_vs_law import USER_AND_OFFICER_ACTION_VS_LAW_PROMPT
from prompts_store.ask_gpt.officer_other_vs_law import OFFICER_ACTION_PLUS_OTHER_VS_LAW_PROMPT
from prompts_store.ask_gpt.user_officer_other_vs_law import USER_AND_OFFICER_ACTION_PLUS_OTHER_VS_LAW_PROMPT

from prompts import PromptBuilder
from constants import SEPARATOR_A, SEPARATOR_A_, SEPARATOR_B, SEPARATOR_B_, QUESTION

st.logger.get_logger = logging.getLogger
st.logger.setup_formatter = '%(asctime)s | %(levelname)s | %(message)s'
st.logger.update_formatter = lambda *a, **k: None
st.logger.set_log_level = lambda *a, **k: None


@st.cache(allow_output_mutation=True)
def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # check if logs.log exists and check the number of lines
    log_files = os.listdir('logs/')
    # get last log file
    if len(log_files) > 0:
        last_log_file = sorted(log_files)[-1]
        with open(f'logs/{last_log_file}', 'r') as f:
            lines = f.readlines()
            if len(lines) > 1000:
                # create a new log file
                log_file = f'logs/logs_{len(log_files)}.log'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            else:
                # use the last log file
                file_handler = logging.FileHandler(f'logs/{last_log_file}')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
    else:
        file_handler = logging.FileHandler('logs/logs.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    
    streamlit_handler = logging.getLogger("streamlit")
    streamlit_handler.setLevel(logging.INFO)
    streamlit_handler.addHandler(file_handler)
    streamlit_handler.addHandler(stdout_handler)

logging.info("Starting Streamlit app")
# streamlit_handler
# we read two different datasets and embeddings
dataset_names = ['leydetransitov4.csv', 'leyseguridadpublicav1.csv']
df_index_columns = ['ley','articulo', 'parte']

embeddings_names = ['embeddings_transito_v4.csv', 'embeddings_seguridad_publica_v1.csv']
embedding_index_columns = ['ley','articulo', 'parte']


def load_sidebar():
    st.sidebar.header("Como funciona:")
    st.sidebar.markdown("TransitoGPT utiliza un modelo de inteligencia artificial para responder preguntas relacionadas a las leyes de tránsito. La aplicación utiliza a la inteligencia artificial para encontrar las secciones de la o las leyes que son más relevantes para tu pregunta y las muestra en la aplicación, después, utiliza las secciones más relevantes para generar una posible respuesta.")

    st.sidebar.header("Leyes incluidas")
    st.sidebar.markdown("""
    - La ley de tránsito del estado de Sonora  
    - La ley de seguridad pública del estado de Sonora
    - _Bando de gobierno de policía municipal_ (en construcción)
    - _Ley policía preventiva y de transito_ (en construcción)

    """)
    # st.sidebar.markdown("La ley de tránsito tiene por objeto regular el tránsito de vehículos y establecer las normas a las que se sujetarán sus conductores y ocupantes, así como los peatones en el estado de Sonora.")
    # st.sidebar.markdown("La ley de seguridad pública tiene  por  objeto determinar  las  instancias  encargadas  de  la  seguridad  pública  en  la  entidad,  regular la integración,  organización  y  funcionamiento  del  Sistema  Estatal  de  Seguridad  Pública, así  como  establecer  las  bases  de  coordinación  entre  el  Estado  y  los  municipios  en  la materia, a fin de integrar y regular la correspondencia de aquel con el Sistema Nacional de Seguridad Pública.")

    space(2)
    # add a section separator
    st.sidebar.markdown("---")
    st.sidebar.markdown("""<small>*Descargo de responsabilidad*  
    Este no es un servicio de asesoría legal. Los resultados de esta aplicación no deben ser tomados como asesoría legal. Si tienes una pregunta legal, debes consultar a un abogado.
    </small>""", unsafe_allow_html=True)

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

def format_section(sections: list[tuple[tuple[str, str, str], tuple[str]]]):
    """Formats a section of the law to be displayed in the Streamlit app.
    Section is a list of tuples like 
    [(('ley de transito', 'articulo 1', 'parte_01'), ('texto del articulo')),..]

    Returns a string with the section formatted as follows:
    "Ley de transito - Articulo: 1": {texto del articulo}
    ...
    """
    full_text = ""
    for section in sections:
        # get the text of the section
        law = section[0][0]
        article = section[0][1]
        section_text = section[1]

        # format the section
        formated_text = f"""
        {law.upper()} - {article.upper()}
        
        ...{section_text}...\n\n
        """
        # append the section to the full text
        full_text += formated_text
    return full_text

    # load the data, and set the index to the title, chapter, and article
    # cache the data but do not show the loading message
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data():
    leydetransito_df = load_dataset(dataset_names[0], df_index_columns, store_path='./datasets_store')
    leyseguridadpublica_df = load_dataset(dataset_names[1], df_index_columns, store_path='./datasets_store')

    leydetransito_embeddings = load_embedding(embeddings_names[0], embedding_index_columns, store_path='./embeddings_store')
    leyseguridadpublica_embeddings = load_embedding(embeddings_names[1], embedding_index_columns, store_path='./embeddings_store')
    return leydetransito_df, leyseguridadpublica_df , leydetransito_embeddings, leyseguridadpublica_embeddings
    

if __name__ == "__main__":
    
    load_sidebar()
    # load the embeddings
    leydetransito_df, leyseguridadpublica_df, leydetransito_embeddings, leyseguridadpublica_embeddings = load_data()
    

    st.title("TránsitoGPT")
    st.header("Ármate de información para enfrentar una situación de tránsito en Sonora.")
    st.markdown("<small>Sigue a [@RubenAMtz](https://twitter.com/RubenAMtz) en Twitter para actualizaciones sobre este proyecto.</small>", unsafe_allow_html=True)
    
    space(2)

    query = st.text_input("Escribe tu pregunta o situación", placeholder=f"{QUESTION}")
    

    if (st.button("Preguntar") and query) or query:
        st.spinner("Buscando la mejor respuesta...")
        
        logging.info(f"Query: {query}")
        # create a prompt and ask to run entity recognition
        ner_prompt_builder = PromptBuilder(QUERY_ENTITY_RECOGNITION_PROMPT)
        ner_prompt = ner_prompt_builder.build_prompt({'section_01': {'text': query, 'max_length': 512, 'separator': SEPARATOR_B}})

        # create a prompt to classify the intent of the question
        intent_prompt_builder = PromptBuilder(INTENT_PROMPT)
        intent_prompt = intent_prompt_builder.build_prompt({'section_01': {'text': query, 'max_length': 512, 'separator': SEPARATOR_B}})

        responses = asyncio.run(run_ask_gpt_async([ner_prompt, intent_prompt], False))
        ner, intent = [parse_answer(res) for res in responses]

        logging.info(f"NER: {ner}")
        logging.info(f"Intent: {intent}")

        if intent['user_action_vs_law'] and not intent['officer_action_vs_law'] and not intent['complaint']:
    
            LDT_relevant_docs_user_action = {}
            res = "No se encontró información sobre esta situación."
            if ner['user_action']:
                # pick the prompt template
                user_action_vs_law_prompt_builder = PromptBuilder(USER_ACTION_VS_LAW_PROMPT)
                # we get the most relevant sections for the user action
                LDT_relevant_docs_user_action = user_action_vs_law_prompt_builder.get_relevant_documents(
                    ner['user_action'], 
                    leydetransito_embeddings, 
                    leydetransito_df
                )
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
                user_action_vs_law_res = ask_gpt(user_action_vs_law_prompt, False)
                logging.info(f"User action vs law: {user_action_vs_law_res}")
                res = user_action_vs_law_res
            


        elif intent['officer_action_vs_law'] and not intent['user_action_vs_law'] and not intent['complaint']:
            
            LDT_relevant_docs_user_action = {}
            LSP_relevant_docs_user_action = {}
            if ner['officer_action_or_consequence']:
                # we get the most relevant sections, officer's actions have to be checked in both datasets
                officer_action_vs_law_prompt_builder = PromptBuilder(OFFICER_ACTION_VS_LAW_PROMPT)
                LDT_relevant_docs_officer_action = officer_action_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'], 
                    leydetransito_embeddings, 
                    leydetransito_df
                )
                LSP_relevant_doc_officer_action = officer_action_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'],
                    leyseguridadpublica_embeddings,
                    leyseguridadpublica_df
                )
                # fill out the prompt 'form'
                officer_action_vs_law_prompt = officer_action_vs_law_prompt_builder.build_prompt({
                        'officer_action_or_consequence': {
                            'text': ner['officer_action_or_consequence'],
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                        'LDT_relevant_docs_officer_action': {
                            'text': list(LDT_relevant_docs_officer_action.values()),
                            'max_length': 1280,
                            'separator': SEPARATOR_B
                        },
                        'LSP_relevant_doc_officer_action': {
                            'text': list(LSP_relevant_doc_officer_action.values()),
                            'max_length': 1280,
                            'separator': SEPARATOR_B
                        }
                    })
                # ask gpt to generate the answer
                officer_action_vs_law_res = ask_gpt(officer_action_vs_law_prompt, False)
                logging.info(f"Officer action vs law: {officer_action_vs_law_res}")
                res = officer_action_vs_law_res


        # both user and officer actions are checked
        elif intent['officer_action_vs_law'] and intent['user_action_vs_law'] and not intent['complaint']:
            
            LDT_relevant_docs_officer_action = {}
            LSP_relevant_doc_officer_action = {}
            LDT_relevant_docs_user_action = {}
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
            if ner['user_action']:
                LDT_relevant_docs_user_action = user_and_officer_vs_law_prompt_builder.get_relevant_documents(
                    ner['user_action'],
                    leydetransito_embeddings,
                    leydetransito_df
                )

            if ner['user_action'] or ner['officer_action_or_consequence']:
                # fill out the prompt 'form'
                user_and_officer_action_vs_law_prompt = user_and_officer_vs_law_prompt_builder.build_prompt({
                    'user_action': {
                        'text': ner['user_action'],
                        'max_length': 256,
                        'separator': SEPARATOR_B_
                    },
                    'LDT_relevant_docs_user_action': {
                        'text': list(LDT_relevant_docs_user_action.values()), 
                        'max_length': 900, 
                        'separator': SEPARATOR_B
                    },
                    'officer_action_or_consequence': {
                        'text': ner['officer_action_or_consequence'],
                        'max_length': 256,
                        'separator': SEPARATOR_B_
                    },
                    'LDT_relevant_docs_officer_action': {
                        'text': list(LDT_relevant_docs_officer_action.values()),
                        'max_length': 900,
                        'separator': SEPARATOR_B
                    },
                    'LSP_relevant_doc_officer_action': {
                        'text': list(LSP_relevant_doc_officer_action.values()),
                        'max_length': 900,
                        'separator': SEPARATOR_B
                    }
                }) 

                    # ask gpt to generate the answer
                user_and_officer_action_vs_law_res = ask_gpt(user_and_officer_action_vs_law_prompt, False)
                logging.info(f"User and officer action vs law: {user_and_officer_action_vs_law_res}")
                res = user_and_officer_action_vs_law_res


        # TODO: maybe change intent['complaint'] to intent['other'] 
        elif intent['officer_action_vs_law'] and intent['user_action_vs_law'] and intent['complaint']:
            
            LDT_relevant_docs_officer_action = {}
            LSP_relevant_doc_officer_action = {}
            LDT_relevant_docs_user_action = {}
            LSP_relevant_doc_complaint = {}
            # we get the most relevant sections, officer's actions have to be checked in both datasets
            if ner['officer_action_or_consequence']:
                user_and_officer_plus_context_vs_law_prompt_builder = PromptBuilder(USER_AND_OFFICER_ACTION_PLUS_OTHER_VS_LAW_PROMPT)
                LDT_relevant_docs_officer_action = user_and_officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'], 
                    leydetransito_embeddings, 
                    leydetransito_df
                )
                LSP_relevant_doc_officer_action = user_and_officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'],
                    leyseguridadpublica_embeddings,
                    leyseguridadpublica_df
                )
            # user's action is checked only in the traffic law dataset
            if ner['user_action']:
                LDT_relevant_docs_user_action = user_and_officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['user_action'],
                    leydetransito_embeddings,
                    leydetransito_df
                )
            # complaint is checked only in the public security law dataset
            if ner['other']:
                LSP_relevant_doc_complaint = user_and_officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['other'],
                    leyseguridadpublica_embeddings,
                    leyseguridadpublica_df
                )
            
            # fill out the prompt 'form'
            if ner['officer_action_or_consequence'] and ner['user_action'] and ner['other']:
                user_and_officer_plus_context_vs_law_prompt_prompt = user_and_officer_plus_context_vs_law_prompt_builder.build_prompt({
                        'user_action': {
                            'text': ner['user_action'] or 'No hay acción del usuario',
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                        'LDT_relevant_docs_user_action': {
                            'text': list(LDT_relevant_docs_user_action.values()), 
                            'max_length': 900, 
                            'separator': SEPARATOR_B
                        },
                        'officer_action_or_consequence': {
                            'text': ner['officer_action_or_consequence'] or 'No hay acción del agente',
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                        'LDT_relevant_docs_officer_action': {
                            'text': list(LDT_relevant_docs_officer_action.values()),
                            'max_length': 900,
                            'separator': SEPARATOR_B
                        },
                        'LSP_relevant_doc_officer_action': {
                            'text': list(LSP_relevant_doc_officer_action.values()),
                            'max_length': 900,
                            'separator': SEPARATOR_B
                        },
                        'other': {
                            'text': ner['other'] or 'No hay contexto adicional',
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                    })

                # ask gpt to generate the answer
                user_and_officer_plus_context_vs_law_prompt_res = ask_gpt(user_and_officer_plus_context_vs_law_prompt_prompt, False)
                logging.info(f"User and officer action plus context vs law: {user_and_officer_plus_context_vs_law_prompt_res}")
                res = user_and_officer_plus_context_vs_law_prompt_res

        elif intent['officer_action_vs_law'] and not intent['user_action_vs_law'] and intent['complaint']:
            
            LDT_relevant_docs_officer_action = {}
            LSP_relevant_doc_officer_action = {}
            LSP_relevant_doc_complaint = {}
            # we get the most relevant sections, officer's actions have to be checked in both datasets
            if ner['officer_action_or_consequence']:
                officer_plus_context_vs_law_prompt_builder = PromptBuilder(OFFICER_ACTION_PLUS_OTHER_VS_LAW_PROMPT)
                LDT_relevant_docs_officer_action = officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'], 
                    leydetransito_embeddings, 
                    leydetransito_df
                )
                LSP_relevant_doc_officer_action = officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['officer_action_or_consequence'],
                    leyseguridadpublica_embeddings,
                    leyseguridadpublica_df
                )
            # complaint is checked only in the public security law dataset
            if ner['other']:
                LSP_relevant_doc_complaint = officer_plus_context_vs_law_prompt_builder.get_relevant_documents(
                    ner['other'],
                    leyseguridadpublica_embeddings,
                    leyseguridadpublica_df
                )
            
            # fill out the prompt 'form'
            if ner['officer_action_or_consequence'] and ner['other']:
                officer_plus_context_vs_law_prompt_prompt = officer_plus_context_vs_law_prompt_builder.build_prompt({
                        'officer_action_or_consequence': {
                            'text': ner['officer_action_or_consequence'] or 'No hay acción del agente',
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                        'LDT_relevant_docs_officer_action': {
                            'text': list(LDT_relevant_docs_officer_action.values()),
                            'max_length': 900,
                            'separator': SEPARATOR_B
                        },
                        'LSP_relevant_doc_officer_action': {
                            'text': list(LSP_relevant_doc_officer_action.values()),
                            'max_length': 900,
                            'separator': SEPARATOR_B
                        },
                        'other': {
                            'text': ner['other'] or 'No hay contexto adicional',
                            'max_length': 256,
                            'separator': SEPARATOR_B_
                        },
                    })

                # ask gpt to generate the answer
                officer_plus_context_vs_law_prompt_res = ask_gpt(officer_plus_context_vs_law_prompt_prompt, False)
                logging.info(f"User and officer action plus context vs law: {officer_plus_context_vs_law_prompt_res}")
                res = officer_plus_context_vs_law_prompt_res

        
        # show a section with the results
        st.markdown("### Resultados")
        st.markdown(f"{res}")
        st.markdown("Te recuerdo que estos resultados son solo una guía, no son una interpretación legal. Si quieres saber más sobre tu caso, puedes consultar a un abogado.")
        
        st.markdown("### Referencias")

        section_text = ""
        if 'user_action_vs_law_prompt_builder' in locals():
            section_text += format_section(user_action_vs_law_prompt_builder.chosen_documents)
        if 'officer_action_vs_law_prompt_builder' in locals():
            logging.info(f"Chosen docs: {officer_action_vs_law_prompt_builder.chosen_documents}")
            section_text += format_section(officer_action_vs_law_prompt_builder.chosen_documents)
        if 'user_and_officer_vs_law_prompt_builder' in locals():
            logging.info(f"Chosen docs: {user_and_officer_vs_law_prompt_builder.chosen_documents}")
            section_text += format_section(user_and_officer_vs_law_prompt_builder.chosen_documents)
        if 'user_and_officer_plus_context_vs_law_prompt_builder' in locals():
            logging.info(f"Chosen docs: {user_and_officer_plus_context_vs_law_prompt_builder.chosen_documents}")
            section_text += format_section(user_and_officer_plus_context_vs_law_prompt_builder.chosen_documents)
        if 'officer_plus_context_vs_law_prompt_builder' in locals():
            logging.info(f"Chosen docs: {officer_plus_context_vs_law_prompt_builder.chosen_documents}")
            section_text += format_section(officer_plus_context_vs_law_prompt_builder.chosen_documents)

        st.markdown(f"{section_text}")
    
    
        if res:
            logging.info("Generating report")
            pdf = PDF()
            title = 'Ármate de información para enfrentar una situación de tránsito'
            pdf.set_title(title)
            pdf.set_author('Ruben Alvarez')
            pdf.front_page()
            pdf.query(query)
            if 'user_action_vs_law_prompt_builder' in locals():
                pdf.print_chapter(user_action_vs_law_prompt_builder.chosen_documents_indexes, [leydetransito_df, leyseguridadpublica_df])
            if 'officer_action_vs_law_prompt_builder' in locals():
                pdf.print_chapter(officer_action_vs_law_prompt_builder.chosen_documents_indexes, [leydetransito_df, leyseguridadpublica_df])
            if 'user_and_officer_vs_law_prompt_builder' in locals():
                pdf.print_chapter(user_and_officer_vs_law_prompt_builder.chosen_documents_indexes, [leydetransito_df, leyseguridadpublica_df])
            if 'user_and_officer_plus_context_vs_law_prompt_builder' in locals():
                pdf.print_chapter(user_and_officer_plus_context_vs_law_prompt_builder.chosen_documents_indexes, [leydetransito_df, leyseguridadpublica_df])
            if 'officer_plus_context_vs_law_prompt_builder' in locals():
                pdf.print_chapter(officer_plus_context_vs_law_prompt_builder.chosen_documents_indexes, [leydetransito_df, leyseguridadpublica_df])

            # now in local time
            # date = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            # save pdf to file object and encode it to base64
            import io
            import base64
            pdf_bytes = io.BytesIO()
            pdf.output(pdf_bytes)
            # # pdf.buffer is a bytearray buffer object
            # base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode('utf-8')
            # # display pdf in the browser
            # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="1000" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)
            
            logging.info("Report generated")

            space(2)
            download_button = st.download_button(
                label="Descargar reporte",
                data=pdf_bytes,
                file_name=f"{title}.pdf",
                mime="application/pdf"
            )

            if download_button:
                logging.info("Report downloaded")
