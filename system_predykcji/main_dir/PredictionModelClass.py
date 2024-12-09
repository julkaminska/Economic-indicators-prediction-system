import json
import os
import pickle
import streamlit as st

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fpdf import FPDF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class PredictionModel():
    def __init__(self, indicator, method):
        self.indicator = indicator
        self.method = method
        self.dataset = 'main_dir/processed_data/indicators.csv'
        self.window_size = self.assign_window_size()
        self.model = self.assign_model()
        self.indicator_data = self.get_indicator_data()
        self.summary = None
        self.forecast_file = f"main_dir/pickles/forecasts/{self.indicator}_{self.method}_forecast.csv"
        if self.method == "nn": self.history_file = f"main_dir/pickles/history/nn_{self.indicator}.json"
        if self.method == "nn": self.scaler = self.load_scaler()
        if not os.path.exists(self.forecast_file): self.initial_forecast()

    def load_scaler(self):
        file = f"main_dir/pickles/scalers/scaler_{self.method}_{self.indicator}.pkl"
        with open(file, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    def assign_window_size(self):
        if self.method == 'arima':
            if self.indicator =='gdp':
                return 4
            else: return 12
        elif self.method == 'nn':
            if self.indicator =='gdp':
                return 4
            else: return 3

    def assign_model(self):
        file = f"main_dir/pickles/{self.method}_{self.indicator}.pkl"
        # if self.method == 'arima':
        #     file = f"pickles/{self.method}_{self.indicator}_2_1_2.pkl"
        # else:
        #     file = f"pickles/{self.method}_{self.indicator}.pkl"

        with open(file, "rb") as f:
            model = pickle.load(f)
        return model

    def save_forecast_csv(self, forecast):
        df = pd.DataFrame({
            "value": np.round(forecast.values, 3),
        }, index=pd.to_datetime(forecast.index))
        df.index.name = 'datetime'
        df.to_csv(self.forecast_file, index=True)
    def load_forecast(self, n_steps):
        if os.path.exists(self.forecast_file):
            with open(self.forecast_file, 'r') as f:
                forecast_df = pd.read_csv(f, index_col=0)
                forecast_df.index = pd.to_datetime(forecast_df.index)
                forecast_values = forecast_df['value'].head(n_steps)
        return forecast_values
    def arima_forecast(self, n_steps, initial_forecast = False):
        if initial_forecast:
            if self.indicator == 'gdp':
                full_steps = 24
                last_historical = self.indicator_data.index[-1] + pd.DateOffset(months=3)
                # date_range = pd.date_range(start=last_historical, periods=full_steps, freq='QS')
            else:
                full_steps = 72
                last_historical = self.indicator_data.index[-1] + pd.DateOffset(months=1)
                # date_range = pd.date_range(start=last_historical, periods=full_steps, freq='MS')
            forecast = self.model.get_forecast(steps=full_steps)
            forecast_values = forecast.predicted_mean
            self.save_forecast_csv(forecast_values)
        return self.load_forecast(n_steps)

    def initial_forecast(self):
        if self.indicator == 'gdp':
            full_steps = 24
        else:
            full_steps = 72
        if self.method == 'arima':
            self.arima_forecast(full_steps, initial_forecast = True)
        elif self.method == 'nn':
            self.nn_forecast(full_steps,initial_forecast = True)

    def nn_forecast(self, n_steps, initial_forecast):
        if initial_forecast:
            if self.indicator == 'gdp':
                full_steps = 24
                last_historical = self.indicator_data.index[-1] + pd.DateOffset(months=3)
                date_range = pd.date_range(start=last_historical, periods=full_steps, freq='QS')
            else:
                full_steps = 72
                last_historical = self.indicator_data.index[-1] + pd.DateOffset(months=1)
                date_range = pd.date_range(start=last_historical, periods=full_steps, freq='MS')
            predictions = []
            scaled_data = self.scaler.fit_transform(self.indicator_data.values)
            current_input = scaled_data[-self.window_size:]
            for _ in range(full_steps):
                input_reshaped = current_input.reshape(1, self.window_size, 1)
                pred = self.model.predict(input_reshaped, verbose=0)[0, 0]
                predictions.append(pred)
                current_input = np.append(current_input[1:], pred)
            predictions = np.array(predictions).reshape(-1, 1)
            inverse_predictions = self.scaler.inverse_transform(predictions)
            forecast_series = pd.Series(inverse_predictions.flatten(), index=date_range, name="Forecast")
            self.save_forecast_csv(forecast_series)
        return self.load_forecast(n_steps)

    def forecast(self, n_steps, time_unit = 'M'):
        if os.path.exists(self.forecast_file):
            if time_unit == "Q" and self.indicator != "gdp":
                n_steps = n_steps * 3
            if time_unit == "Y" and self.indicator != "gdp":
                n_steps = n_steps * 12
            if time_unit == "Y" and self.indicator == "gdp":
                n_steps = n_steps * 4
            if time_unit == "M" and self.indicator == "gdp":
                n_steps = n_steps % 3
            return self.load_forecast(n_steps)
        else:
            if self.model:
                if self.method == 'arima':
                    return self.arima_forecast(n_steps, initial_forecast=True)
                elif self.method == 'nn':
                    return self.nn_forecast(n_steps, initial_forecast=True)

    def plot_descriptions(self):
        description = {
            'gdp': ['Produkt krajowy brutto (PKB) w Polsce', 'PKB [mln zł]'],
            'inflation': ['Poziom inflacji w Polsce<br>Wskaźnik CPI przy podstawie analogicznego miesiąca poprzedniego roku [%]',
                'Wskaźnik CPI [%]'],
            'unemployment': ['Stopa bezrobocia rejestrowanego w Polsce', 'Stopa bezrobocia [%]']}
        return description[self.indicator]

    def get_indicator_data(self):
        df = pd.read_csv(self.dataset, index_col = 0)
        indicator_data = df[[f"{self.indicator}"]].dropna()
        if self.indicator == 'gdp':
            indicator_data.index.freq = 'QS'
        else:
            indicator_data.index.freq = 'MS'
        indicator_data.rename(columns={self.indicator: 'value'}, inplace=True)
        indicator_data.index = pd.to_datetime(indicator_data.index)
        return indicator_data

    def arima_params(self):
        y = self.indicator_data
        fitted_values = self.model.fittedvalues

        non_zero_id = y.value != 0
        non_zero_y = y[non_zero_id]
        forecast_non_zero = fitted_values[non_zero_id]
        mape = round(np.mean(np.abs((non_zero_y.value - forecast_non_zero) / non_zero_y.value) * 100),2)


        p = self.model.model_orders['ar']
        q = self.model.model_orders['ma']
        d = self.model.model_orders['variance']

        aic = round(self.model.aic, 2)
        return [p, d, q, aic, mape]

    def format_params(self, params):
        if self.method == 'arima':
            parameters = [f"- parametr autoregresyjny (p): {params[0]}",
                    f"-	rzad roznicowania (d): {params[1]}",
                    f"- parametr sredniej ruchomej (q): {params[2]}"]
            metrics = [f"Kryterium informacyjne Akaikego (AIC):{params[3]}",
                    f"Sredni bezwzgledny blad procentowy (MAPE): {params[4]}%"]

        elif self.method == 'nn':
            parameters = [f"Warstwy sieci:",
                          f"{'; '.join(str(line) for line in params[0])}",
                          f"Optymalizator: {params[1]}",
                          f"Wspolczynnik uczenia: {params[2]}"]
            metrics = [f" MSE na zbiorze treningowym: {params[3]}",
                       f" MSE na zbiorze testowym: {params[4]}"]
        return parameters, metrics
    def nn_params(self):
        metrics_file = f"main_dir/pickles/metrics/{self.method}_{self.indicator}.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            layers = metrics['layers']
            optimizer = metrics['optimizer']
            lr = metrics["lr"]
        with open(self.history_file, 'r') as f1:
            history = json.load(f1)
            mse = history['mse'][-1]
            val_mse = history['val_mse'][-1]
            formatted_mse = "{:.2e}".format(mse)
            formatted_val_mse = "{:.2e}".format(val_mse)
        return [layers, optimizer, lr, formatted_mse, formatted_val_mse]

    def model_summary(self):
        if self.method == 'arima':
            params = self.arima_params()
        elif self.method == 'nn':
            params = self.nn_params()
        return params