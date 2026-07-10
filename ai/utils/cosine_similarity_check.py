import numpy as np

# -----------------------------------------
# Cosine Similarity
# -----------------------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))