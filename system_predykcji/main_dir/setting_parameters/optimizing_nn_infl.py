import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from keras_tuner import Hyperband
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, TimeSeriesSplit


def get_indicator_data(dataset_path):
    indicators = ['gdp', 'inflation', 'unemployment']
    indicators_dataframes = []
    df = pd.read_csv(dataset_path, index_col=0)
    for indicator in indicators:
        indicator_data = df[[f"{indicator}"]].dropna()
        indicator_data.rename(columns={indicator: 'value'}, inplace=True)
        indicator_data.index = pd.to_datetime(indicator_data.index)
        if indicator == 'gdp':
            indicator_data.index.freq = 'QS'
        else:
            indicator_data.index.freq = 'MS'
        indicators_dataframes.append(indicator_data)
    return indicators_dataframes


data = '../processed_data/indicators.csv'
indicators_data = get_indicator_data(data)
gdp_data, inflation_data, unemployment_data = indicators_data[0], indicators_data[1], indicators_data[2]
series = inflation_data.values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series)
with open('../pickles/scalers/scaler_nn_inflation.pkl', 'wb') as f:
    pickle.dump(scaler, f)

def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 12
X, y = create_windowed_data(scaled_series, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', min_value=1, max_value=3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16), activation='relu'))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def search_best_params():
    tuner = Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=100,
        directory='tuning_dir',
        project_name='tuning_mlp',
        max_consecutive_failed_trials=8,
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32,
                     callbacks=[early_stopping])
    best_model = tuner.get_best_models(1)[0]

    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Rzeczywiste wartości")
    plt.plot(y_pred, label="Prognozowane wartości")
    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.legend()
    plt.show()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    for key, value in best_hps.values.items():
        print(f"{key}: {value}")

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    print(history.history)
    val_acc_per_epoch = history.history['mae']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
    # eval_result = hypermodel.evaluate(X_val, y_val)
    hypermodel.summary()
    #
    with open("../pickles/nn_inflation.pkl", 'wb') as file:
        pickle.dump(hypermodel, file)
    #
    with open('../pickles/history/nn_inflation.json', 'w') as f:
        json.dump(history.history, f)

    # Wykresy błędu treningu i walidacji
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def infl_model():
    window_size = 12
    X, y = create_windowed_data(scaled_series, window_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        Dense(12, activation = 'relu'),
        Dense(80, activation = 'relu'),
        Dense(32, activation = 'relu'),
        Dense(1)

    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    with open("../pickles/nn_inflation.pkl", 'wb') as file:
        pickle.dump(model, file)

    with open('../pickles/history/nn_inflation.json', 'w') as f:
        json.dump(history.history, f)

    # Wykres strat
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Błąd na zbiorze treningowym')
    plt.plot(history.history['val_loss'], label='Błąd na zbiorze walidacyjnym')
    plt.legend()
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.show()

    # Wykres dopasowania
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    time = np.arange(len(scaled_series))
    train_time = np.arange(len(train_pred))
    test_time = np.arange(len(train_pred), len(train_pred) + len(test_pred))  # Czas dla danych testowych
    plt.figure(figsize=(12, 6))
    plt.plot(time, scaled_series, label="Rzeczywiste wartości")
    plt.plot(train_time, train_pred[:, 0, 0], label="Przewidywania na zbiorze treningowym")
    plt.plot(test_time, test_pred[:, 0, 0], label="Przewidywania na zbiorze testowym")
    plt.suptitle('Dopasowanie modelu do danych - Dane dot. poziomu inflacji')
    mse = history.history['val_mse'][-1]
    formatted_mse = "{:.2e}".format(mse)
    plt. title(f'MSE:{formatted_mse}')
    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.legend()
    plt.show()

    # Histogram błędów
    errors = y_test[:, 0] - test_pred[:, 0, 0]
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=10, color='skyblue', edgecolor='black')
    plt.suptitle("Histogram błędu")
    plt.title('Poziom inflacji')
    plt.xlabel("Wartość błędu")
    plt.ylabel("Liczba próbek")
    plt.show()
    return model

def predict_future(model, data, steps, window_size):
    predictions = []
    current_input = data[-window_size:]
    for _ in range(steps):
        input_reshaped = current_input.reshape(1, window_size, 1)
        pred = model.predict(input_reshaped, verbose=0)[0,0]
        predictions.append(pred)
        current_input = np.append(current_input[1:], pred)
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)
# Funkcja do prognozowania


def generate_initial_files(model = None):
    summary = {}
    layers = []
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        layer_info = f"Warstwa {i+1}: L. neuronow: {config.get('units', '0')}, Funkcja aktywacji: {config.get('activation', 'Brak')}"
        layers.append(layer_info)
    optimizer = str(type(model.optimizer)).split('.')[-1].strip("'>")
    summary['layers'] = layers
    summary['optimizer'] = optimizer
    summary['lr'] = round(float(model.optimizer.learning_rate.numpy()),3)

    with open('../pickles/metrics/nn_inflation.json', 'w') as f:
        json.dump(summary, f)

    forecast_file = "../pickles/forecasts/inflation_nn_forecast.csv"

    future_steps = 72
    future_predictions = predict_future(model, scaled_series, future_steps, window_size)

    last_historical = inflation_data.index[-1] + pd.DateOffset(months=1)
    date_range = pd.date_range(start=last_historical, periods=future_steps, freq='MS')


    df = pd.DataFrame({
                "value": np.round(future_predictions.flatten(),1),
            }, index = date_range)
    df.index.name = 'datetime'
    df.to_csv(forecast_file, index=True)

    history = scaler.inverse_transform(scaled_series).flatten()
    predicted_time_range = range(len(history), len(history) + future_steps)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history))[-72:], history[-72:], label="Historyczne wartości", color='blue')
    plt.plot(predicted_time_range, future_predictions.flatten(), label="Prognozowane wartości", color='red')
    plt.axvline(x=len(history) - 1, linestyle='--', color='gray', label="Punkt startowy prognozy")
    plt.title("Historyczne i prognozowane wartości")
    plt.xlabel("Czas")
    plt.ylabel("Wartość wskaźnika")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history)), history, label="Historyczne wartości", color='blue')
    plt.plot(predicted_time_range, future_predictions.flatten(), label="Prognozowane wartości", color='red')
    plt.axvline(x=len(history) - 1, linestyle='--', color='gray', label="Punkt startowy prognozy")
    plt.title("Historyczne i prognozowane wartości")
    plt.xlabel("Czas")
    plt.ylabel("Wartość wskaźnika")
    plt.legend()
    plt.grid()
    plt.show()

    print("Prognozowane wartości:", future_predictions.flatten())

def plot():
    forecast_file = "../pickles/forecasts/inflation_nn_forecast.csv"

    future_steps = 72
    last_historical = inflation_data.dropna().index[-1] + pd.DateOffset(months=1)
    date_range = pd.date_range(start=last_historical, periods=future_steps, freq='MS')
    with open(forecast_file, 'r') as f:
        forecast_df = pd.read_csv(f, index_col=0)
        forecast_df.index = pd.to_datetime(forecast_df.index)
        forecast_values = forecast_df['value']

    plt.figure(figsize=(12, 6))
    plt.plot(inflation_data.dropna().index[-73:], inflation_data[-73:], label="Historyczne wartości", color='blue')
    plt.plot(date_range, forecast_values, label="Prognozowane wartości", color='red')
    plt.axvline(x=inflation_data.dropna().index[-1], linestyle='--', color='gray', label="Punkt startowy prognozy")
    plt.suptitle("Historyczne i prognozowane wartości - dane dot. poziomu inflacji")
    plt.title("Model sieci neuronowej")
    plt.xlabel("Czas")
    plt.ylabel("Wartość wskaźnika")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(inflation_data.dropna().index, inflation_data, label="Historyczne wartości", color='blue')
    plt.plot(date_range, forecast_values, label="Prognozowane wartości", color='red')
    plt.axvline(x=inflation_data.dropna().index[-1], linestyle='--', color='gray', label="Punkt startowy prognozy")
    plt.suptitle("Historyczne i prognozowane wartości - dane dot. poziomu inflacji")
    plt.title("Model sieci neuronowej")
    plt.xlabel("Czas")
    plt.ylabel("Wartość wskaźnika")
    plt.legend()
    plt.grid()
    plt.show()

# search_best_params()
# model = infl_model()
# generate_initial_files(model)
plot()