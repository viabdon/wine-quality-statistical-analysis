import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    """
    ## Args:
        None.

    ## Description:
        Carrega e combina datasets de vinho

    ## Returns:
        pd.DataFrame
    """
    red_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        sep=";",
    )
    white_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";",
    )

    red_wine["type"] = "red"
    white_wine["type"] = "white"

    df = pd.concat([red_wine, white_wine], ignore_index=True)
    return df


def apply_log_transforms(df):
    """
    ## Args:
        pd.DataFrame

    ## Description:
        Aplica transformações logarítmicas em variáveis assimétricas

    ## Returns:
        pd.DataFrame
    """

    df_transformed = df.copy()

    # Transformando apenas as que melhoraram conforme visto na EDA
    log_vars = ["residual sugar", "chlorides", "sulphates"]

    for var in log_vars:
        df_transformed[f"{var} log"] = np.log1p(df[var])

    return df_transformed


def prepare_features_red(df):
    """
    ## Args:
        pd.DataFrame

    ## Description:
        Prepara features para regressão linear do vinho tinto baseado na análise de correlação

    ## Returns:
        `X`: pd.DataFrame, `y`: pd.DataFrame
    """

    df_red = df[df["type"] == "red"].copy()

    # Removendo as colunas redundantes identificadas na EDA
    features_to_drop = [
        "citric acid",  # Correlação forte com fixed acidity
        "pH",  # Correlação forte negativa com acidez
        "free sulfur dioxide",  # Manter apenas total sulfur dioxide
        "type",  # Não é numérica
    ]

    # Usaremos apenas as versões log das variáveis transformadas, então removemos as originais
    X = df_red.drop(
        columns=features_to_drop
        + ["quality", "residual sugar", "chlorides", "sulphates"]
    )

    # Adicionndo versões log
    if "residual sugar log" in df_red.columns:
        X["residual sugar log"] = df_red["residual sugar log"]
        X["chlorides log"] = df_red["chlorides log"]
        X["sulphates log"] = df_red["sulphates log"]

    # Separando a variável alvo
    y = df_red["quality"]

    return X, y


def prepare_features_white(df):
    """    
    ## Args:
        pd.DataFrame

    ## Description:
        Prepara features para regressão linear do vinho branco baseado na análise de correlação

    ## Returns:
        `X`: pd.DataFrame, `y`: pd.DataFrame
    """
    
    df_white = df[df["type"] == "white"].copy()

    # Removendo densidade (redundante com álcool e açúcar)
    features_to_drop = [
        "density",  # Função de álcool e açúcar
        "free sulfur dioxide",  # Manter apenas total
        "type" # Não é numérica,
    ]

     # Mesma lógica aplicada para o vinho tinto
    X = df_white.drop(
        columns=features_to_drop
        + ["quality", "residual sugar", "chlorides", "sulphates"]
    )

    # Mesma lógica aplicada para o vinho tinto
    if "residual sugar_log" in df_white.columns:
        X["residual sugar_log"] = df_white["residual sugar_log"]
        X["chlorides_log"] = df_white["chlorides_log"]
        X["sulphates_log"] = df_white["sulphates_log"]

    # Separando a variável alvo
    y = df_white["quality"]

    return X, y


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    ## Args:
        `X`: pd.DataFrame 
        *Features*
        `y`: pd.DataFrame 
        *Target*
        
    ## Description:
        Divide dados em treino/validação/teste
    
    ## Returns:
        `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` 
        *Dataframes de teste e de treino*    
    """
    
    # Primeiro split: treino+val vs teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Segundo split: treino vs validação
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_baseline(y_train, y_val, y_test):
    """
    ## Args:
        `y_train`: pd.DataFrame 
        `y_val`: pd.DataFrame 
        `y_test`: pd.DataFrame 
        
    ## Description:
        Calcula baseline usando média de treino
    ## Returns:
        `baseline_metrics`: dict
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
    baseline_pred = np.full(len(y_test), y_train.mean())

    baseline_metrics = {
        "MSE": mean_squared_error(y_test, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "R²": r2_score(y_test, baseline_pred),
    }

    return baseline_metrics


if __name__ == "__main__":
    # Teste rápido
    df = load_data()
    df = apply_log_transforms(df)

    print("Red Wine:")
    X_red, y_red = prepare_features_red(df)
    print(f"Features: {X_red.columns.tolist()}")
    print(f"Shape: {X_red.shape}")

    print("\nWhite Wine:")
    X_white, y_white = prepare_features_white(df)
    print(f"Features: {X_white.columns.tolist()}")
    print(f"Shape: {X_white.shape}")
