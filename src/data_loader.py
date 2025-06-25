import pandas as pd


def load_dataset(fake_path: str, real_path: str) -> tuple:
    """
    Carrega o dataset das notícias falsas e reais a partir de arquivos CSV.

    Args:
        fake_path (str): Caminho para o arquivo CSV contendo o dataset de notícias falsas.
        real_path (str): Caminho para o arquivo CSV contendo o dataset de notícias reais.

    Returns:
        tuple: Uma tupla contendo 3 pandas DataFrames, um para o dataset de notícias falsas, outro para o dataset de notícias reais
        e outro para o dataset concatenado.
    """
    
    # Carrega os datasets de notícias falsas e reais
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    
    # Adiciona uma coluna 'label' para identificar o tipo de notícia, sendo 0 = Fake e 1 = Real
    fake_df['label'] = 0
    real_df['label'] = 1
    
    # Concatena os DataFrames de notícias falsas e reais
    concat_df = pd.concat([fake_df, real_df], ignore_index=True)
    
    return (fake_df, real_df, concat_df)
