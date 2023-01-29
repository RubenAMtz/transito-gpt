import streamlit as st
import pandas as pd
import numpy as np
import openai
import tiktoken
import core
import constants

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

def format_section(sections: list[tuple[float, tuple[str, str, str]]], df: pd.DataFrame):
    """Formats a section of the law to be displayed in the Streamlit app.
    Section is a list of tuples like (0.8596269121097068, (' SEGUNDO ', ' V', '81'))

    Returns a string with the section formatted as follows:
    "Articulo: 81": {text of the section according to df}
    "Articulo: 82": {text of the section according to df}
    ....
    """
    full_text = ""
    for section in sections:
        # get the text of the section
        section_text = core.get_section_text(section, df)
        # get the title, chapter, and article of the section
        article = section[1]
        # format the section
        formated_text = f"""
        {article.upper()}
        
        {section_text}\n\n
        """
        # append the section to the full text
        full_text += formated_text
    return full_text

    

if __name__ == "__main__":
    # load the embeddings
    
    # load the data, and set the index to the title, chapter, and article
    # cache the data but do not show the loading message
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def load_data():
        df = pd.read_csv("leydetransitov2.csv", header=0)
        # df = df.set_index(['titulo','capitulo','articulo'])
        df = df.set_index(['articulo'])

        document_embeddings = core.load_embeddings("embeddingsv2.csv")
        return df, document_embeddings
    
    df, document_embeddings = load_data()
    

    st.title("TransitoGPT")
    st.markdown("##### TransitoGPT es una aplicación que te ayuda a entender la ley de transito de Sonora.")
    st.markdown("Sigue a [@RubenAMtz](https://twitter.com/RubenAMtz) en Twitter para actualizaciones sobre este proyecto.")
    # add double space
    space(2)

    st.markdown("### Describe tu situación o realiza una pregunta")
    query = st.text_input("Escribe tu pregunta o situación", value="¿Qué pasa si me detienen por manejar ebrio?")

    st.markdown("#### Ley de transito")
    state = st.radio("¿En qué estado?", options=["Sonora"])

    if st.button("Preguntar"):
        # get most relevant sections, three is enough
        sections = core.order_document_sections_by_query_similarity(query, document_embeddings)[:3]

        # construct a prompt for the query using the most relevant sections as context
        prompt = core.construct_prompt(query, document_embeddings, df)
        # call the API with the prompt
        answer = core.answer_query_with_context(query, df, document_embeddings)
        
        # show a section with the results
        st.markdown("### Resultados")
        st.markdown(f"{answer}")
        
        st.markdown("### Referencias")
        section_text = format_section(sections, df)
        st.markdown(f"{section_text}")
    
    space(2)
    # add a section separator
    st.markdown("---")
    st.markdown("##### Descargo de responsabilidad")
    st.markdown("Este no es un servicio de asesoría legal. Los resultados de esta aplicación no deben ser tomados como asesoría legal. Si tienes una pregunta legal, debes consultar a un abogado.")
    




