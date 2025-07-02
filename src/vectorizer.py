from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer() -> TfidfVectorizer:
    """
    Cria um vetorizador TF-IDF com stopwords em inglês.
        
    Returns:
        TfidfVectorizer: Um vetor TF-IDF configurado.
    """
    
    return TfidfVectorizer(
        stop_words="english",
    )
    