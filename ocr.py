from pdfminer.high_level import extract_text
import json
# save it in csv format using pandas
import pandas as pd

og_text = extract_text("docs/leydetransito.pdf")

# DOCUMENT has the following hierarchical structure:
# TITULO
## CAPITULO
### SECCION
#### ARTICULO
##### a), b), c) .... n)
#### (señales has a special section SP, S..., at the same level of ARTICULO) 

pdf = {}

# leave 'T R A N S I T O R I O S' out (from page 62 onwards), this also leaves 'A P E N D I C E' and 'I N D I C E' out
text = og_text.split('T R A N S I T O R I O S')[0]

# split by TITULO, leave anything before the first TITULO out.
titulos = text.split('TITULO')

# initialize pdf
for titulo in titulos[1:]:
    titulo_temp = titulo.split('\n')[0]
    pdf[titulo_temp] = {}
    
# loop through titulos
for titulo in titulos[1:]:
    titulo_temp = titulo.split('\n')[0]
    
    # split by CAPITULO
    capitulos = titulo.split('CAPITULO')
    
    # loop through capitulos
    for capitulo in capitulos[1:]:
        capitulo_temp = capitulo.split('\n')[0]
        pdf[titulo_temp][capitulo_temp] = {}

        # split by SECCION
        secciones = capitulo.split('SECCION')
        
        # loop through secciones
        for seccion in secciones[1:]:
            seccion_temp = seccion.split('\n')[0]
            pdf[titulo_temp][capitulo_temp][seccion_temp] = {}
            
            # split by ARTICULO
            articulos = seccion.split('ARTICULO')
            
            # loop through articulos
            for articulo in articulos[1:]:
                articulo_temp = articulo.split('.-')[0]
                pdf[titulo_temp][capitulo_temp][seccion_temp][articulo_temp.strip()] = articulo.split('.-')[1]



# in case 'SECCION' is missing
for titulo in titulos[1:]:
    titulo_temp = titulo.split('\n')[0]
    capitulos = titulo.split('CAPITULO')
    
    for capitulo in capitulos[1:]:
        capitulo_temp = capitulo.split('\n')[0]
        
        if not 'SECCION' in capitulo:
            pdf[titulo_temp][capitulo_temp]['ARTICULO'] = {}
            articulos = capitulo.split('ARTICULO')
            
            for articulo in articulos[1:]:
                articulo_temp = articulo.split('.-')[0]
                pdf[titulo_temp][capitulo_temp]['ARTICULO'][articulo_temp.strip()] = articulo.split('.-')[1]


# dump the dict to a json file
with open('leydetransito.json', 'w', encoding='utf-8') as f:
    json.dump(pdf, f, ensure_ascii=False, indent=4)


# Values have whitespaces at the beginning, middle and end, remove them
with open('leydetransito.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for titulo in data:
    for capitulo in data[titulo]:
        for seccion in data[titulo][capitulo]:
            for articulo in data[titulo][capitulo][seccion]:
                data[titulo][capitulo][seccion][articulo] = data[titulo][capitulo][seccion][articulo].strip()
                # remove multiple whitespaces
                data[titulo][capitulo][seccion][articulo] = ' '.join(data[titulo][capitulo][seccion][articulo].split())



# dump the dict to a json file
with open('leydetransito.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


df = pd.DataFrame(columns=['titulo', 'capitulo', 'seccion', 'articulo', 'texto'])

for titulo in data:
    for capitulo in data[titulo]:
        for seccion in data[titulo][capitulo]:
            for articulo in data[titulo][capitulo][seccion]:
                df = df.append({'titulo': titulo, 'capitulo': capitulo, 'seccion': seccion, 'articulo': articulo, 'texto': data[titulo][capitulo][seccion][articulo]}, ignore_index=True)

df.to_csv('leydetransito.csv', index=False)