import numpy as np
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def max_sum_similarity(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int,
    nr_candidates: int,
) -> List[Tuple[str, float]]:
    """Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and
    extract the combination that are the least similar to each other
    by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large `top_n`

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception(
            "Make sure that the number of candidates exceeds the number "
            "of keywords to return."
        )

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum(
            [candidates[i][j] for i in combination for j in combination if i != j]
        )
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [
        (words_vals[idx], round(float(distances[0][words_idx[idx]]), 4))
        for idx in candidate
    ]
    
def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]