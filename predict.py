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
