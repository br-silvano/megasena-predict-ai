import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Carregar novos dados
new_data = pd.read_csv('data/megasena_2024.csv')

# Preprocessamento
X_new = new_data.drop(columns=['date']).values
y_new = new_data.drop(columns=['date']).values

# Escalar os dados
scaler = joblib.load('model/scaler.pkl')
X_new_scaled = scaler.transform(X_new)
y_new_scaled = scaler.transform(y_new)

# Preparar dados para LSTM
X_new_scaled = np.array([X_new_scaled[i:i+1]
                        for i in range(len(X_new_scaled))])
y_new_scaled = np.array([y_new_scaled[i] for i in range(len(y_new_scaled))])

# Carregar modelo existente
model = tf.keras.models.load_model('model/model.h5')

# Treinar o modelo com os novos dados
model.fit(X_new_scaled, y_new_scaled, epochs=50, batch_size=32, verbose=1)

# Salvar modelo atualizado
model.save('model/updated_model.h5')
