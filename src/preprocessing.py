import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download de recursos necessários do NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Inicialização de stopwords e lemmatizer
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Realiza limpeza e pré-processamento do texto, incluindo:
    - Conversão para minúsculas
    - Remove URLS, dígitos, pontuações, caracteres especiais e stopwords
    - Lematiza os tokens
    
    Args:
        text (str): Texto a ser limpo e pré-processado.
    
    Returns:
        str: Texto limpo e pré-processado.
    """
    
    # Verifica se o texto é NaN e retorna uma string vazia
    if pd.isna(text):
        return ""
    
    # Converte o texto para minúsculas
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove dígitos
    text = re.sub(r'\d+', '', text)
    
    # Remove pontuações e caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove uma "assinatura" ou marca de fonte para notícias verdadeiras em ambos os datasets
    text = re.sub(r'\(?Reuters\)?|"Reuters"|REUTERS', '', text, flags=re.IGNORECASE)
    
    # Remove espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenização do texto
    tokens = text.split()
    
    # Remove stopwords e lemmatiza os tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords]

    # Junta os tokens de volta em uma string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pré-processamento de texto em uma coluna específica do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contendo a coluna 'text' a ser pré-processada.
    
    Returns:
        pd.DataFrame: DataFrame com a coluna 'text' pré-processada.
    """
    # Evita modificar o DataFrame original
    df = df.copy()
    
    # Preenche valores nulos nas colunas do DataFrame
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['subject'] = df['subject'].fillna('Unknown')
    
    # Cria coluna full_text combinando 'title' e 'text'
    df['full_text'] = df['title'] + ' ' + df['text']
    
    # Aplica a função de limpeza de texto na coluna 'full_text'
    df['full_text_clean'] = df['full_text'].apply(clean_text)
    
    return df