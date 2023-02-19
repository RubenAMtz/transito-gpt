from core import save_embeddings, compute_doc_embeddings
import pandas as pd
import time
import os

ocr_data = 'leyseguridadpublicav1.csv'
name = 'embeddings_seguridad_publica_v1.csv'
index_columns = ['articulo', 'parte']

# load the csv file
df = pd.read_csv(ocr_data)
# define the index columns
df = df.set_index(index_columns)

document_embeddings = None
if name not in os.listdir():
    # compute the embeddings for 20 rows at a time
    rpm = 20  # requests per minute
    embeddings = {}
    for i in range(0, len(df), rpm):
        print(f"Computing embeddings for rows {i} to {i + rpm}...")
        # compute the embeddings for the next 20 rows
        embeddings.update(compute_doc_embeddings(df.iloc[i:i + rpm]))
        # break
        time.sleep(60)
    # add the number of tokens per row to the embedding based on the text

    # save the embeddings to a file
    save_embeddings(f"{name}", index_columns, embeddings, store_path='embeddings_store')

else:
    print("Embeddings file already computed.")