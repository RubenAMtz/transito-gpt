from pdfminer.high_level import extract_text
import json
import pandas as pd
import re
import tiktoken

gpt_encoding = tiktoken.get_encoding('p50k_base') # encoding for text-davinci-003
og_text = extract_text("docs/leydetransito.pdf")
name = 'leydetransitov3' # change this to the name of the csv and json files you want to save

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


# initialize pdf
pdf = {}
# loop through articulos and print the number of tokens
for i, articulo in enumerate(articulos):
    lenght = len(gpt_encoding.encode(articulo))
    encoded_article = gpt_encoding.encode(articulo)
    pdf[f'articulo {i+1}'] = {}
    split_len = 300
    stride = 150
    if lenght > split_len:
        # iterate through the encoded article with a sliding window of 300 tokens and stride of 150 tokens
        for j in range(0, lenght, stride):
            if j+split_len <= lenght:
                pdf[f'articulo {i+1}'][f'parte_{j//stride+1:02}'] = gpt_encoding.decode(encoded_article[j:j+split_len])
            else:
                pdf[f'articulo {i+1}'][f'parte_{j//stride+1:02}'] = gpt_encoding.decode(encoded_article[j:])
    else:
        pdf[f'articulo {i+1}']['parte_01'] = articulo

# save pdf as json, encoding='utf-8' is important to avoid errors
with open(f'{name}.json', 'w', encoding='utf-8') as f:
    json.dump(pdf, f, ensure_ascii=False, indent=4)

# create a dataframe with the following structure:
# articulo | parte | texto
df = pd.DataFrame(columns=['articulo', 'parte', 'texto'])
for articulo, partes in pdf.items():
    for parte, texto in partes.items():
        # append is about to be depricated, use concat instead
        df = pd.concat([df, pd.DataFrame([[articulo, parte, texto]], columns=['articulo', 'parte', 'texto'])])

# add a column named "tokens" with the number of tokens in each articulo, spaces are also considered tokens
df['tokens'] = df['texto'].apply(lambda x: len(gpt_encoding.encode(x)))

df.columns = ['articulo', 'parte', 'texto', 'tokens']
df.to_csv(f'{name}.csv', encoding='utf-8')
