### Prevendo Números de Loteria com Redes Neurais Recorrentes (RNNs) e LSTM em Python

A previsão de números de loteria, como a Mega-Sena no Brasil, é um desafio intrigante e complexo devido à natureza aleatória dos sorteios. Embora a precisão dessas previsões seja inerentemente limitada, podemos explorar técnicas avançadas de aprendizado de máquina, como Redes Neurais Recorrentes (RNNs) e Long Short-Term Memory (LSTM), para tentar identificar padrões e tendências nos dados históricos. Este artigo oferece uma abordagem prática para criar, treinar, prever e avaliar um modelo de RNN/LSTM usando Python.

#### Coleta e Pré-processamento dos Dados

Para começar, assumimos que temos um arquivo `megasena.csv` contendo os dados históricos dos sorteios da Mega-Sena:

```csv
date,n1,n2,n3,n4,n5,n6
2024-07-25,46,55,6,31,52,26
2024-07-23,15,24,44,40,4,47
2024-07-20,42,18,53,13,52,4
2024-07-18,11,7,36,52,19,12
2024-07-16,27,38,43,8,44,25
2024-07-13,19,52,46,50,43,32
...
```

#### Treinando o Modelo com RNN/LSTM

Usaremos uma Rede Neural Recorrente com LSTM para tentar prever os números da loteria. O código a seguir treina o modelo e o salva para uso posterior.

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import numpy as np
import pandas as pd

# Carregar dados
data = pd.read_csv('data/megasena.csv')

# Preprocessamento
X = data.drop(columns=['date']).values
y = data.drop(columns=['date']).values

# Escalar os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.transform(y)

# Preparar dados para LSTM
X_scaled = np.array([X_scaled[i:i+1] for i in range(len(X_scaled))])
y_scaled = np.array([y_scaled[i] for i in range(len(y_scaled))])

# Definir K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Parâmetros
epochs = 200
batch_size = 32
validation_split = 0.1

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Resultados
val_losses = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_scaled[train_index], y_scaled[val_index]

    # Definir o modelo LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 6)))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse')

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=1)

    # Guardar os resultados de validação
    val_loss = min(history.history['val_loss'])
    val_losses.append(val_loss)

# Treinar o modelo final com todo o conjunto de treinamento
model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=batch_size,
          callbacks=[early_stopping], verbose=1)

# Salvar modelo e scaler
model.save('model/model.h5')
joblib.dump(scaler, 'model/scaler.pkl')

print(
    f"Modelo treinado e salvo com sucesso! Loss de validação: {np.mean(val_losses):.4f}")
```

#### Prevendo Números com o Modelo Treinado

Após treinar o modelo, podemos usá-lo para prever os números da Mega-Sena.

```python
import joblib
import numpy as np
import tensorflow as tf

# Carregar modelo e scaler
model = tf.keras.models.load_model('model/model.h5')
scaler = joblib.load('model/scaler.pkl')


# Prever números
def predict_numbers_with_percentages():
    # Neste exemplo, usamos números aleatórios como entrada
    # Pode-se modificar para entrada personalizada
    input_data = np.random.randint(1, 61, size=(1, 6))
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = np.array([input_data_scaled])

    prediction_scaled = model.predict(input_data_scaled)

    # Exibir percentuais para cada número
    percentages = prediction_scaled[0]

    predicted_numbers = scaler.inverse_transform(prediction_scaled)
    predicted_numbers = np.clip(predicted_numbers, 1, 60).astype(int)

    # Mapear os percentuais para os números previstos
    results = list(zip(predicted_numbers[0], percentages))

    return results


predict_numbers_with_percentages = predict_numbers_with_percentages()

print("Número | Percentual")
print("--------------------")
for number, percentage in predict_numbers_with_percentages:
    print(f"{number:2d}     | {percentage:.2f}")
```

#### Avaliando a Eficiência do Modelo

A avaliação do modelo é crucial para entender seu desempenho. Utilizamos várias métricas para uma análise completa.

```python
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
```

#### Métricas de Avaliação

1. **Mean Squared Error (MSE)**: Média dos quadrados dos erros. Penaliza mais os grandes erros.
2. **Mean Absolute Error (MAE)**: Média dos valores absolutos dos erros. Penaliza todos os erros igualmente.
3. **F1 Score**: Média harmônica de precisão e recall, fornece uma medida balanceada.

### Conclusão

Embora a previsão precisa dos números da loteria seja extremamente desafiadora devido à sua natureza aleatória, este artigo demonstrou como usar técnicas avançadas de aprendizado de máquina, como RNNs e LSTMs, para tentar identificar padrões nos dados históricos. Utilizamos várias métricas para avaliar a eficácia do modelo e geramos relatórios e gráficos para uma análise visual detalhada. Apesar das limitações, essa abordagem oferece uma base sólida para explorar o uso de aprendizado de máquina em problemas complexos e aleatórios.

---

### Referências

1. **TensorFlow e Keras**: Frameworks usados para construir e treinar o modelo de rede neural.
2. **Scikit-learn**: Biblioteca usada para preprocessamento de dados e cálculo de métricas.
3. **Matplotlib e Seaborn**: Bibliotecas utilizadas para visualização de dados e gráficos.

---

### Código Completo

Os códigos fornecidos neste artigo podem ser encontrados nos arquivos:
- `train.py`
- `predict.py`
- `evaluate.py`

Esses arquivos contêm todas as etapas detalhadas para coleta, treino, previsão e avaliação do modelo.
