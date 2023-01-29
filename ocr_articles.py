from pdfminer.high_level import extract_text
import json
import pandas as pd
import re
import tiktoken

encoding = tiktoken.get_encoding('p50k_base') # encoding for text-davinci-003
og_text = extract_text("docs/leydetransito.pdf")

# create a dictionary with the following hierarchical structure:
"""
pdf ={
    "articulo 01": {
        "parte_01": "texto del articulo 01",
        "parte_02": "texto del articulo 01",
        },
    "articulo 02": {
        "parte_01": "texto del articulo 02"
        },
    ...
}
"""
# remove all \n
og_text = og_text.replace('\n', ' ')
og_text = re.sub(' +', ' ', og_text)

# remove page number, we find it because it appears before '\x0c', use regex to find it
og_text = re.sub(r'\d+ \x0c', '\x0c', og_text)
# remove all '\x0c'
og_text = og_text.replace('\x0c', ' ')


# split document by articulo, use regex to split by 'ARTICULO 1.-', 'ARTICULO 2.-', etc. there are also cases like 'ARTICULO *1.-', 'ARTICULO *2.-', etc.
# find all the matches of 'ARTICULO \d+.-' and 'ARTICULO \*\d+.-'
import re
matches = re.finditer(r'ARTICULO \d+.-|ARTICULO \*\d+.- | ARTICULO \d+o.- | ARTICULO \*\d+o.-', og_text)
# get the start and end positions of each match
positions = [(m.start(0), m.end(0)) for m in matches]
# split the document by the positions
articulos = [og_text[i:j] for i, j in zip([0]+[j for i, j in positions], [i for i, j in positions]+[None])]
# remove the first element, which is non relevant
articulos = articulos[1:]

# remove whitespaces from the beginning and end of each articulo
articulos = [articulo.strip() for articulo in articulos]


# loop through articulos and print the number of tokens
for i, articulo in enumerate(articulos):
    lenght = len(encoding.encode(articulo))
    if lenght > 8190:
        print(f'articulo {i+1} has {lenght} tokens')
    if i == 197:
        print(f'articulo {i+1} has {lenght} tokens')
    

# initialize pdf
pdf = {}
for i, articulo in enumerate(articulos):
    pdf[f'articulo {i+1}'] = articulo

# save pdf as json, encoding='utf-8' is important to avoid errors
with open('leydetransitov2.json', 'w', encoding='utf-8') as f:
    json.dump(pdf, f, ensure_ascii=False, indent=4)

# save pdf as csv, encoding='utf-8' is important to avoid errors
# key = articulo, value = texto, but assume value is another column
df = pd.DataFrame.from_dict(pdf, orient='index').reset_index()
# add a column named "tokens" with the number of tokens in each articulo, spaces are also considered tokens

df['tokens'] = df[0].apply(lambda x: len(encoding.encode(x)))

df.columns = ['articulo', 'texto', 'tokens']
df.to_csv('leydetransitov2.csv', encoding='utf-8')
