from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import numpy as np
import pandas as pd

# Carregar dados
data = pd.read_csv('data/megasena_train.csv')

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
