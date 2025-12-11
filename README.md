mode# Projeto de Modelagem Estatística – Wine Quality

Este repositório contém o desenvolvimento do Projeto do 2º Bimestre da disciplina
**Modelagem Estatística** (CESUPA).

O objetivo é aplicar:

- Análise exploratória dos dados (EDA)
- Limpeza e preparação dos dados
- Regressão linear simples, múltipla e polinomial usando **statsmodels** e **sklearn** para interpretação pipelines e avaliação
- Classificação com **Naive Bayes** e **Regressão Logística**
- Divisão em treino/validação/teste
- Métricas adequadas para cada tipo de modelo
- Validação cruzada e tuning (GridSearc)

Dataset escolhido: **Wine Quality (Vinho Tinto e Branco)**  
Fonte: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/186/wine+quality
Licença: Creative Commons (CC BY 4.0)

---

## ⚙️ Estrutura do Repositório
```
wine-quality-statistical-analysis/
│
├── data/
│   ├── wine+quality.zip # Arquivo com os dados brutos para caso caia a base de dados online
│
├── notebooks/
│   └── projeto_modelagem.ipynb   # notebook final com EDA + modelagem + tuning
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py       # limpeza, splits, baseline
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## ▶ Como executar (terminal)
### Observação: **utilize python 3.10 ou 3.11**
1. Clone o repositório:

```
git clone https://github.com/viabdon/wine-quality-statistical-analysis.git

cd seu-projeto-wine-quality
```

2. Instale as dependências:

```
pip install -r requirements.txt
```

3. Execute o Jupyter Notebook:
```
jupyter notebook
```

Abra o arquivo:  
`notebooks/projeto.ipynb`

---

## 📚 Referências

- UCI Machine Learning Repository – Wine Quality Dataset  
- Statsmodels Documentation  
- Scikit-Learn Documentation  
