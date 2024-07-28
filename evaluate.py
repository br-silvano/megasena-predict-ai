from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Carregar dados
data = pd.read_csv('data/megasena_2024.csv')

# Preprocessamento
X = data.drop(columns=['date']).values
y = data.drop(columns=['date']).values

# Escalar os dados
scaler = joblib.load('model/scaler.pkl')
X_scaled = scaler.transform(X)
y_scaled = scaler.transform(y)

# Preparar dados para LSTM
X_scaled = np.array([X_scaled[i:i+1] for i in range(len(X_scaled))])
y_scaled = np.array([y_scaled[i] for i in range(len(y_scaled))])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# Carregar modelo
model = tf.keras.models.load_model('model/model.h5')

# Avaliar modelo
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test = scaler.inverse_transform(y_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Criar relatório
metrics = {
    'MSE': mse,
    'MAE': mae,
    'R2 Score': r2
}

# Converter para DataFrame
metrics_df = pd.DataFrame(metrics, index=[0])

# Salvar relatório em CSV
metrics_df.to_csv('report/evaluation_report.csv', index=False)

# Plotar métricas
metrics_melted = metrics_df.melt(var_name='Metric', value_name='Value')

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x='Metric', y='Value')
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.xticks(rotation=45)
plt.tight_layout()

# Salvar gráfico
plt.savefig('report/evaluation_metrics.png')
plt.show()
