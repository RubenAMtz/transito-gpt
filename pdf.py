from fpdf import FPDF, HTMLMixin
from typing import Any

import pandas as pd

class PDF(FPDF, HTMLMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.add_font('Times', '', 'C:/Windows/Fonts/times.ttf', uni=True)
        self.add_font('PoppinsM', '', 'fonts/Poppins-Medium.ttf', uni=True)
        self.add_font('PoppinsL', '', 'fonts/Poppins-Light.ttf', uni=True)

    def header(self):
        # quick fix:
        if self.page_no() != 1:
            return
        # Arial bold 15
        self.set_font('PoppinsM', '', 20)
        # Calculate width of title and position
        w = self.get_string_width(self.title) + 6
        # self.set_x((110 - w) / 2)
        # Title
        self.multi_cell(0, None, self.title, 0, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Disclaimer
        self.multi_cell(0, None, f'Este no es un servicio de asesoría legal. Los resultados de esta aplicación no deben ser tomados como asesoría legal. Si tienes una pregunta legal, debes consultar a un abogado.\n\nPágina {str(self.page_no())}', 0, 'C')
        # self.cell(0,0, f'', 0, 0, 'C')
        # Page number
        # self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def front_page(self):
        self.add_page()
        # image in the center of the page
        with self.round_clip(80, 120, 50):
            self.image('logo.png', 80, 120, 50, link='https://transito-gpt.streamlit.app/')
        self.set_font('PoppinsM', '', 25)
        self.ln(130)
        self.cell(0, 10, 'TránsitoGPT', 0, 1, 'C')

    def query(self, query):
        self.add_page()
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, None, f'El usuario preguntó: {query}', 0, 'C')
        self.ln(10)

    def chapter_title(self, label, num):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, ' %s : %s' % (label, num), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, document_text):
        self.set_font('PoppinsL', '', 12)
        # Output justified text
        self.multi_cell(0, 5, document_text)
        # Line break
        self.ln()
        # Mention in italics
        self.set_font('Arial', 'I')
        self.cell(0, 5, '(final del extracto)')
        # Line break
        self.ln(10)


    def print_chapter(self, chosen_documents, datasets: list[pd.DataFrame]):
        # merge the datasets into a single list
        dataset = pd.concat(datasets, axis=0)
        dataset.reset_index(inplace=True)
        dataset.set_index(['ley', 'articulo', 'parte'], inplace=True)
        # load chosen documents, indexes are of the form 
        # [(law, article, part), (law, article, part), ...]
        # we only care about [(law, article), (law, article), ...)], there are duplicates, keep them all
        for document in chosen_documents:
            law, article, part = document
            # filter articles (smaller set than law)
            res = dataset.filter(like=article, axis=0)
            # filter by law
            res = res.filter(like=law, axis=0)
            text = " ".join(res.texto.values)
            # self.add_page()
            self.chapter_title(law.capitalize() , article.capitalize())
            self.chapter_body(text)

    def print_chapter_from_text(self, text):
        self.set_font('Arial', '', 12)
        # Output justified text
        self.multi_cell(0, 5, text)
        # Line break
        self.ln()