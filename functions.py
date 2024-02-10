import PyPDF2
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

def read_pdf_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    # Eliminar los caracteres de salto de línea ("\n")
    text = text.replace("\n", "")
    return text


def create_dataframe_from_text(text):
    # El problema puede estar aquí, si el texto es una lista, se puede unir usando un espacio
    if isinstance(text, list):
        text = " ".join(text)

    # Dividir el texto en partes relevantes (aquí es solo un ejemplo)
    lines = text.split("\n")  # Dividir el texto en líneas

    # Crear un diccionario con los datos
    data = {"Texto": lines}

    # Crear DataFrame de pandas
    df = pd.DataFrame(data)

    return df


def LDA_view(documents):
    # Tokenización de los documentos
    texts = [[word for word in document.lower().split()] for document in documents]

    # Crear un diccionario de términos
    dictionary = corpora.Dictionary(texts)

    # Convertir los documentos en bolsas de palabras
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Entrenar el modelo LDA
    lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary)

    # Visualizar el modelo LDA
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda_visualization.html')  