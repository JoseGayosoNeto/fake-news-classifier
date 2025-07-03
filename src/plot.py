from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


def plot_top_words_by_class(df, text_col, label_col, label_value, top_n=20):
    """
    Plota as palavras mais frequentes de um DataFrame filtrado por rótulo.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        text_col (str): Nome da coluna com o texto.
        label_col (str): Nome da coluna com os rótulos.
        label_value (int): Valor do rótulo para filtrar (0 para Falsas, 1 para Verdadeiras).
        top_n (int): Número de palavras mais frequentes a serem exibidas.
    
    Returns:
        None: Exibe um gráfico de barras com as palavras mais frequentes.
    """
    # Filtra pelo rótulo desejado
    subset = df[df[label_col] == label_value]
    
    # Cria o vetorizador
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(subset[text_col])
    
    # Soma ocorrências por termo
    word_counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    
    # DataFrame com as frequências
    freq_df = pd.DataFrame({'word': vocab, 'count': word_counts})
    freq_df = freq_df.sort_values(by='count', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(data=freq_df, x='count', y='word', palette='viridis')
    plt.title(f'Top {top_n} palavras - Classe {"Falsa" if label_value == 0 else "Verdadeira"}')
    plt.xlabel('Frequência')
    plt.ylabel('Palavra')
    plt.show()

def plot_top_bigrams_by_class(df, text_col, label_col, label_value, ngram_range=(2,2), top_n=20):
    """
    Plota os bi-grams mais frequentes de um DataFrame filtrado por rótulo.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        text_col (str): Nome da coluna com o texto.
        label_col (str): Nome da coluna com os rótulos.
        label_value (int): Valor do rótulo para filtrar (0 para Falsas, 1 para Verdadeiras).
        ngram_range (tuple): Intervalo de n-grams a serem considerados (ex: (2,2) para bi-grams).
        top_n (int): Número de n-grams mais frequentes a serem exibidos.
    
    Returns:
        None: Exibe um gráfico de barras com os bi-grams mais frequentes.
    """
    subset = df[df[label_col] == label_value]
    
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(subset[text_col])
    
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    
    freq_df = pd.DataFrame({'ngram': vocab, 'count': counts})
    freq_df = freq_df.sort_values(by='count', ascending=False).head(top_n)
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=freq_df, x='count', y='ngram', palette='rocket')
    plt.title(f'Top {top_n} bi-grams - Classe {"Falsa" if label_value == 0 else "Verdadeira"}')
    plt.xlabel('Frequência')
    plt.ylabel('bi-grama')
    plt.show()

def get_top_tfidf_terms_by_class(df, text_col, label_col, label_value, top_n=20):
    """
    Calcula e plota os termos mais importantes com base no TF-IDF para uma classe específica.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        text_col (str): Nome da coluna com o texto.
        label_col (str): Nome da coluna com os rótulos.
        label_value (int): Valor do rótulo para filtrar (0 para Falsas, 1 para Verdadeiras).
        top_n (int): Número de termos mais importantes a serem exibidos.
    
    Returns:
        None: Exibe um gráfico de barras com os termos mais importantes.
    """
    subset = df[df[label_col] == label_value]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(subset[text_col])
    
    mean_tfidf = X.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    
    tfidf_df = pd.DataFrame({'term': vocab, 'mean_tfidf': mean_tfidf})
    tfidf_df = tfidf_df.sort_values(by='mean_tfidf', ascending=False).head(top_n)
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=tfidf_df, x='mean_tfidf', y='term', palette='mako')
    plt.title(f'Top {top_n} TF-IDF termos médios - Classe {"Falsa" if label_value == 0 else "Verdadeira"}')
    plt.xlabel('TF-IDF médio')
    plt.ylabel('Termo')
    plt.show()

def sentiment_analysis(df):
    """
    Análise de Sentimento usando TextBlob. Calcula a Polaridade e a Subjetividade de cada Classe.
        
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        
    Returns:
        None: Exibe um gráfico mostrando o grau de Polaridade e Subjetividade para cada Classe.
    """
    
    def analyze_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    sentiments = df['full_text_clean'].apply(analyze_sentiment)
    df['polarity'] = sentiments.apply(lambda x: x[0])
    df['subjectivity'] = sentiments.apply(lambda x: x[1])
    
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x='label', y='polarity')
    plt.title('Distribuição da Polaridade por Classe')
    plt.xticks([0, 1], ['Falsas', 'Verdadeiras'])
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x='label', y='subjectivity')
    plt.title('Distribuição da Subjetividade por Classe')
    plt.xticks([0, 1], ['Falsas', 'Verdadeiras'])
    plt.show()


