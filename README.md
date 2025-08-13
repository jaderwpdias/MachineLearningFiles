# Exercícios de Regressão Linear - Machine Learning

Este projeto contém implementações de regressão linear simples e múltipla em diferentes datasets, avaliando métricas de performance e gerando análises visuais.

## Exercícios Incluídos

### Exercício 1: Regressão Linear com Dataset Iris
- **Dataset**: Iris (scikit-learn)
- **Variável alvo**: `petal length (cm)`
- **Features simples**: `sepal length (cm)`
- **Features múltiplas**: `sepal length (cm)`, `sepal width (cm)`, `petal width (cm)`

### Exercício 2: Regressão Linear com Dataset Diabetes
- **Dataset**: Diabetes (scikit-learn)
- **Variável alvo**: Progressão da doença
- **Features simples**: `bmi` (índice de massa corporal)
- **Features múltiplas**: `bmi`, `age`, `sex`

## Objetivos dos Exercícios

- Aplicar regressão linear simples e múltipla em diferentes datasets
- Avaliar métricas de performance: MAE, MSE, RMSE e R²
- Gerar gráficos de análise: Previsto vs Real, Resíduos vs Previsto, e Coeficientes
- Comparar o desempenho entre regressão simples e múltipla
- Experimentar com padronização de dados usando StandardScaler

## Instalação

1. Clone ou baixe este repositório
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

### Exercício 1 - Dataset Iris:
```bash
python regressao_iris.py
```

### Exercício 2 - Dataset Diabetes:
```bash
python regressao_diabetes.py
```

## Estrutura do Projeto

- `regressao_iris.py`: Exercício 1 - Regressão com dataset Iris
- `regressao_diabetes.py`: Exercício 2 - Regressão com dataset Diabetes
- `requirements.txt`: Dependências do projeto
- `README.md`: Este arquivo de documentação
- `images/`: Pasta contendo os gráficos gerados pelos scripts
- `.gitignore`: Configuração para ignorar arquivos desnecessários no Git

## Saídas Geradas

### Exercício 1 (Iris):
- `images/regressao_simples_previsto_vs_real.png`
- `images/regressao_simples_residuos.png`
- `images/regressao_multipla_previsto_vs_real.png`
- `images/regressao_multipla_residuos.png`
- `images/coeficientes_regressao_multipla.png`

### Exercício 2 (Diabetes):
- `images/diabetes_regressao_simples_previsto_vs_real.png`
- `images/diabetes_regressao_simples_residuos.png`
- `images/diabetes_regressao_multipla_previsto_vs_real.png`
- `images/diabetes_regressao_multipla_residuos.png`
- `images/diabetes_coeficientes_regressao_multipla.png`

## Análise dos Resultados

Cada exercício irá mostrar:
1. Métricas de performance para ambos os modelos
2. Comparação entre regressão simples e múltipla
3. Análise dos coeficientes
4. Experimentação com dados padronizados
5. Informações específicas sobre cada dataset

## Divisão dos Dados

- 75% para treino
- 25% para teste
- Random state: 42 (para reprodutibilidade)

## Tags do Git

- `v1.0.0`: Versão inicial com Exercício 1 (Iris)
- `licao-1`: Tag para Lição 1 - Regressão Linear com Iris
- `v2.0.0`: Versão com Exercício 2 (Diabetes)
- `licao-2`: Tag para Lição 2 - Regressão Linear com Diabetes

## Informações sobre os Datasets

### Dataset Iris
- 150 amostras de flores Iris
- 4 variáveis contínuas (sepal length, sepal width, petal length, petal width)
- Dataset clássico para machine learning

### Dataset Diabetes
- 442 pacientes com diabetes
- Variável alvo: medida quantitativa da progressão da doença
- Features: bmi, age, sex, entre outras variáveis médicas
- Dataset comumente usado para demonstrar técnicas de regressão linear
