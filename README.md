# Exercício 1 — Regressão Linear e Múltipla com o Iris

Este projeto implementa regressão linear simples e múltipla no conjunto de dados Iris do scikit-learn, avaliando as métricas MAE, MSE, RMSE e R².

## Objetivos

- Aplicar regressão linear simples e múltipla no dataset Iris
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

Execute o script principal:

```bash
python regressao_iris.py
```

## Estrutura do Projeto

- `regressao_iris.py`: Script principal com toda a implementação
- `requirements.txt`: Dependências do projeto
- `README.md`: Este arquivo de documentação

## Saídas Geradas

O script irá gerar os seguintes arquivos de gráficos:
- `regressao_simples_previsto_vs_real.png`
- `regressao_simples_residuos.png`
- `regressao_multipla_previsto_vs_real.png`
- `regressao_multipla_residuos.png`
- `coeficientes_regressao_multipla.png`

## Análise dos Resultados

O script irá mostrar:
1. Métricas de performance para ambos os modelos
2. Comparação entre regressão simples e múltipla
3. Análise dos coeficientes
4. Experimentação com dados padronizados

## Variáveis Utilizadas

- **Variável alvo**: `petal length (cm)`
- **Regressão Simples**: `sepal length (cm)`
- **Regressão Múltipla**: `sepal length (cm)`, `sepal width (cm)`, `petal width (cm)`

## Divisão dos Dados

- 75% para treino
- 25% para teste
- Random state: 42 (para reprodutibilidade)
