import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Dataset Iris carregado:")
print(df.head())
print(f"\nForma do dataset: {df.shape}")
print(f"Colunas: {list(df.columns)}")

# 2. Definir variável alvo
y = df['petal length (cm)']
print(f"\nVariável alvo: petal length (cm)")
print(f"Estatísticas da variável alvo:")
print(y.describe())

# 3. Criar conjuntos de features
# Regressão Linear Simples
X1 = df[['sepal length (cm)']]

# Regressão Múltipla
Xn = df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]

print(f"\nFeatures para regressão simples: {list(X1.columns)}")
print(f"Features para regressão múltipla: {list(Xn.columns)}")

# 4. Dividir dados em treino e teste
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=42)
Xn_train, Xn_test, _, _ = train_test_split(Xn, y, test_size=0.25, random_state=42)

print(f"\nTamanho dos conjuntos de treino: {X1_train.shape[0]}")
print(f"Tamanho dos conjuntos de teste: {X1_test.shape[0]}")

# 5. Treinar modelos
# Modelo de Regressão Linear Simples
modelo_simples = LinearRegression()
modelo_simples.fit(X1_train, y_train)

# Modelo de Regressão Múltipla
modelo_multipla = LinearRegression()
modelo_multipla.fit(Xn_train, y_train)

# 6. Fazer predições
y_pred_simples = modelo_simples.predict(X1_test)
y_pred_multipla = modelo_multipla.predict(Xn_test)

# 7. Calcular métricas
def calcular_metricas(y_true, y_pred, nome_modelo):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== Métricas para {nome_modelo} ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mae, mse, rmse, r2

metricas_simples = calcular_metricas(y_test, y_pred_simples, "Regressão Linear Simples")
metricas_multipla = calcular_metricas(y_test, y_pred_multipla, "Regressão Múltipla")

# 8. Gerar gráficos
plt.rcParams['figure.figsize'] = (10, 8)

# Gráfico 1: Previsto vs Real - Regressão Simples
plt.figure()
plt.scatter(y_test, y_pred_simples, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Regressão Linear Simples: Previsto vs Real')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regressao_simples_previsto_vs_real.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 2: Resíduos vs Previsto - Regressão Simples
residuos_simples = y_test - y_pred_simples
plt.figure()
plt.scatter(y_pred_simples, residuos_simples, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Regressão Linear Simples: Resíduos vs Previsto')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regressao_simples_residuos.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 3: Previsto vs Real - Regressão Múltipla
plt.figure()
plt.scatter(y_test, y_pred_multipla, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Regressão Múltipla: Previsto vs Real')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regressao_multipla_previsto_vs_real.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 4: Resíduos vs Previsto - Regressão Múltipla
residuos_multipla = y_test - y_pred_multipla
plt.figure()
plt.scatter(y_pred_multipla, residuos_multipla, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Regressão Múltipla: Resíduos vs Previsto')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regressao_multipla_residuos.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 5: Coeficientes da Regressão Múltipla
plt.figure()
features = Xn.columns
coeficientes = modelo_multipla.coef_
plt.bar(features, coeficientes)
plt.xlabel('Features')
plt.ylabel('Coeficientes')
plt.title('Coeficientes da Regressão Múltipla')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coeficientes_regressao_multipla.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Comparação das métricas
print("\n" + "="*50)
print("COMPARAÇÃO ENTRE OS MODELOS")
print("="*50)

print(f"\nRegressão Linear Simples:")
print(f"  MAE: {metricas_simples[0]:.4f}")
print(f"  MSE: {metricas_simples[1]:.4f}")
print(f"  RMSE: {metricas_simples[2]:.4f}")
print(f"  R²: {metricas_simples[3]:.4f}")

print(f"\nRegressão Múltipla:")
print(f"  MAE: {metricas_multipla[0]:.4f}")
print(f"  MSE: {metricas_multipla[1]:.4f}")
print(f"  RMSE: {metricas_multipla[2]:.4f}")
print(f"  R²: {metricas_multipla[3]:.4f}")

# 10. Análise dos resultados
print("\n" + "="*50)
print("ANÁLISE DOS RESULTADOS")
print("="*50)

if metricas_multipla[3] > metricas_simples[3]:
    print("A regressão múltipla apresentou melhor desempenho que a regressão simples.")
    print("Isso ocorre porque a inclusão de mais variáveis (sepal width e petal width)")
    print("permite capturar mais informações sobre a variável alvo petal length.")
    print("O R² mais alto indica que o modelo múltiplo explica melhor a variabilidade dos dados.")
else:
    print("A regressão simples apresentou desempenho similar ou melhor que a múltipla.")
    print("Isso pode indicar que as variáveis adicionais não contribuem significativamente")
    print("para a predição da variável alvo, ou podem estar introduzindo ruído.")

# 11. Experimentação com StandardScaler (Opcional)
print("\n" + "="*50)
print("EXPERIMENTAÇÃO COM PADRONIZAÇÃO (StandardScaler)")
print("="*50)

# Padronizar features
scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X1_train)
X1_test_scaled = scaler.transform(X1_test)
Xn_train_scaled = scaler.fit_transform(Xn_train)
Xn_test_scaled = scaler.transform(Xn_test)

# Treinar modelos com dados padronizados
modelo_simples_scaled = LinearRegression()
modelo_simples_scaled.fit(X1_train_scaled, y_train)

modelo_multipla_scaled = LinearRegression()
modelo_multipla_scaled.fit(Xn_train_scaled, y_train)

# Fazer predições
y_pred_simples_scaled = modelo_simples_scaled.predict(X1_test_scaled)
y_pred_multipla_scaled = modelo_multipla_scaled.predict(Xn_test_scaled)

# Calcular métricas
print("\nMétricas com dados padronizados:")
metricas_simples_scaled = calcular_metricas(y_test, y_pred_simples_scaled, "Regressão Simples (Padronizada)")
metricas_multipla_scaled = calcular_metricas(y_test, y_pred_multipla_scaled, "Regressão Múltipla (Padronizada)")

# Comparar coeficientes
print(f"\nCoeficientes da regressão simples (original): {modelo_simples.coef_[0]:.4f}")
print(f"Coeficientes da regressão simples (padronizada): {modelo_simples_scaled.coef_[0]:.4f}")

print(f"\nCoeficientes da regressão múltipla (original):")
for i, feature in enumerate(Xn.columns):
    print(f"  {feature}: {modelo_multipla.coef_[i]:.4f}")

print(f"\nCoeficientes da regressão múltipla (padronizada):")
for i, feature in enumerate(Xn.columns):
    print(f"  {feature}: {modelo_multipla_scaled.coef_[i]:.4f}")

print("\n" + "="*50)
print("EXPLICAÇÃO SOBRE PADRONIZAÇÃO")
print("="*50)
print("A padronização com StandardScaler transforma os dados para ter média 0 e desvio padrão 1.")
print("Isso faz com que os coeficientes sejam mais comparáveis entre si, pois todas as variáveis")
print("passam a ter a mesma escala. As métricas de performance (MAE, MSE, RMSE, R²) permanecem")
print("as mesmas, pois a padronização não afeta a qualidade da predição, apenas a interpretabilidade")
print("dos coeficientes.")

print("\nExercício concluído com sucesso!")
