import os
import pickle

from ai.scripts.recognize import EMBEDDING_FILE

def load_embeddings():

    if not os.path.exists(EMBEDDING_FILE):
        return []

    with open(EMBEDDING_FILE,"rb") as file:
        return pickle.load(file)


def save_embeddings(data):

    with open(EMBEDDING_FILE,"wb") as file:
        pickle.dump(data, file)