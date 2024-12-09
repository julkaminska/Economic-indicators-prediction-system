import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima


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

def acf_plots(indicator_data, name):
    f = plt.figure()
    f.suptitle(f"Wykres ACF: {name}")

    ax1 = f.add_subplot(511)
    plot_acf(indicator_data.value.dropna(), ax = ax1)

    ax2 = f.add_subplot(513)
    plot_acf(indicator_data.value.diff().dropna(), ax = ax2)

    ax3 = f.add_subplot(515)
    plot_acf(indicator_data.value.diff().diff().dropna(), ax = ax3)

    plt.show()

def acf_pacf_plots(indicator_data, name, d):
    f = plt.figure()
    if d == 1:
        plot_data = indicator_data.value.diff().dropna()
    elif d == 2:
        plot_data = indicator_data.value.diff().diff().dropna()
    else:
        plot_data = indicator_data.value.dropna()

    f.suptitle(f"Wykres ACF i PACF:\n{name}; stopień różnicowania: {d}")
    ax1 = f.add_subplot(412)
    plot_acf(plot_data, ax = ax1)

    ax2 = f.add_subplot(414)
    plot_pacf(indicator_data.value.dropna(), ax=ax2)
    plt.show()


def adfuler_plot(indicator_data):
    result = adfuller(indicator_data.value.dropna())
    print(f"ADF statistics: {result[0]}")
    print("p-value: ", round(result[1],3))
    result = adfuller(indicator_data.value.diff().dropna())
    print(f"ADF statistics, after 1st differencing: {result[0]}")
    print("p-value, after 1st differencing: ", result[1])
    result = adfuller(indicator_data.value.diff().diff().dropna())
    print(f"ADF statistics, after 2nd differencing: {result[0]}")
    print("p-value, after 2nd differencing: ", result[1])
    f = plt.figure()
    ax1 = f.add_subplot(511)
    ax1.set_title("Raw Data")
    ax1.plot(indicator_data.value)
    ax2 = f.add_subplot(513)
    ax2.set_title("1st Order Differencing")
    ax2.plot(indicator_data.value.diff().dropna())
    ax3 = f.add_subplot(515)
    ax3.set_title("2nd Order Differencing")
    ax3.plot(indicator_data.value.diff().diff().dropna())
    plt.show()

def pacf_plot(indicator_data, name):
    f = plt.figure()
    f.suptitle(f"PACF plots: {name}")

    ax1 = f.add_subplot(511)
    plot_pacf(indicator_data.value.dropna(), ax = ax1)

    ax2 = f.add_subplot(513)
    plot_pacf(indicator_data.value.diff().dropna(), ax = ax2)

    ax3 = f.add_subplot(515)
    plot_pacf(indicator_data.value.diff().diff().dropna(), ax = ax3)

    plt.show()

def grid_search_arima(dataset, d):
    model = auto_arima(dataset, seasonal = False, d = d, stepwise = True, trace = True)
    print("Best model parameters:")
    print(model.summary())
    print(model)
    return model

def grid_search_sarima(dataset, seasonality, d=0):
    model = auto_arima(dataset, seasonal=True, d =d, m=seasonality, stepwise=True, trace=True)
    print("Best model parameters:")
    print(model.summary())
    return model

dataset = '../processed_data/indicators.csv'
indicators_data = get_indicator_data(dataset)
gdp_data, inflation_data, unemployment_data = indicators_data[0], indicators_data[1], indicators_data[2]

# INFLACJA
# adfuler_plot(inflation_data)
# acf_plots(inflation_data, "inflation")
# pacf_plot(inflation_data, "iflation")
# acf_pacf_plots(inflation_data,"Poziom inflacji", 1)

# best_infl_0 = grid_search_arima(inflation_data, d = 0 )
# best_infl_1 = grid_search_arima(inflation_data, d = 1 )

infl_model = SARIMAX(endog = inflation_data, order = (3,1,5), seasonal_order = (0,0,0,0))
infl_model_fit = infl_model.fit()
# with open("../pickles/arima_inflation.pkl", 'wb') as file:
#     pickle.dump(infl_model_fit, file)

predicted_infl = infl_model_fit.predict()
forecast = infl_model_fit.get_forecast(steps=72)
forecast_values = forecast.predicted_mean
full_data = pd.concat([inflation_data.value[-72:], forecast_values])
plt.figure(figsize=(12, 6))
plt.plot(full_data.index, full_data, color = 'blue', label = 'Dane historyczne')
plt.plot(forecast_values, color='red', label = 'Wartości prognozowane')
plt.axvline(x=inflation_data.dropna().index[-1], linestyle='--', color='gray', label="Punkt startowy prognozy")
plt.legend()
plt.suptitle("Historyczny i prognozowany przebieg wartości - dane dot. poziomu inflacji")
plt.title(f"Metoda ARIMA")
plt.grid()
plt.show()

# AIC_infl = round(infl_model_fit.aic,2)
# non_zero_infl = inflation_data.value != 0
# non_zero_infl_data = inflation_data[non_zero_infl]
# forecast_non_zero = predicted_infl[non_zero_infl]
# mape_infl = round(np.mean(np.abs((non_zero_infl_data.value - forecast_non_zero) / non_zero_infl_data.value) * 100),2)
#
# plt.plot(inflation_data.index, inflation_data.value, color='red', label='Dane rzeczywiste')
# plt.plot(predicted_infl, color='blue', label = 'Predykcje modelu')
# plt.legend()
# plt.suptitle("Dopasowanie modelu do danych (poziom inflacji)")
# plt.title(f"AIC: {AIC_infl}, MAPE: {mape_infl}%")
# plt.show()

# PKB
# adfuler_plot(gdp_data)
# acf_pacf_plots(gdp_data,"Produkt krajowy brutto  (PKB)", 2)

# acf_plots(gdp_data, "GDP")
# pacf_plot(gdp_data, "GDP")
# best_gdp = grid_search_sarima(gdp_data, 4, d=2)
#
# gdp_model = SARIMAX(endog = gdp_data, order = (3,2,0), seasonal_order = (0,1,2,4))
# gdp_model_fit = gdp_model.fit()
# with open("../pickles/arima_gdp.pkl", 'wb') as file:
#     pickle.dump(gdp_model_fit, file)
# forecast = gdp_model_fit.get_forecast(steps=24)
# forecast_values = forecast.predicted_mean
# full_data = pd.concat([gdp_data.value[-24:], forecast_values])
# plt.figure(figsize=(12, 6))
# plt.plot(full_data.index, full_data, color = 'blue', label = 'Dane historyczne')
# plt.plot(forecast_values, color='red', label = 'Wartości prognozowane')
# plt.axvline(x=unemployment_data.dropna().index[-1], linestyle='--', color='gray', label="Punkt startowy prognozy")
# plt.legend()
# plt.suptitle("Historyczny i prognozowany przebieg wartości - dane dot. PKB")
# plt.title(f"Metoda ARIMA")
# plt.grid()
# plt.show()


# predicted_gdp = gdp_model_fit.predict()
# AIC_gdp = round(gdp_model_fit.aic,2)
# mape_gdp = round(np.mean(np.abs((gdp_data.value - predicted_gdp) / gdp_data.value) * 100),2)
#
# plt.plot(gdp_data.index, gdp_data.value, color='red', label='Dane rzeczywiste')
# plt.plot(predicted_gdp, color='blue', label = 'Predykcje modelu')
# plt.legend()
# plt.suptitle("Dopasowanie modelu do danych (PKB)")
# plt.title(f"AIC: {AIC_gdp}, MAPE: {mape_gdp}%")
#
# plt.show()


# STOPA BEZROBOCIA
# acf_pacf_plots(unemployment_data,"Stopa bezrobocia", 2)
# adfuler_plot(unemployment_data)
# # acf_plots(unemployment_data, "unemployment")
# # pacf_plot(unemployment_data, "unemployment")
# best_unempl = grid_search_sarima(unemployment_data, 12, d=2)
# unempl_model = SARIMAX(endog = unemployment_data, order = (2,2,2), seasonal_order = (1,0,1,12))
# unempl_model_fit = unempl_model.fit()
# with open("../pickles/arima_unemployment.pkl", 'wb') as file:
#     pickle.dump(unempl_model_fit, file)

# forecast = unempl_model_fit.get_forecast(steps=72)
# forecast_values = forecast.predicted_mean
# full_data = pd.concat([unemployment_data.value[-72:], forecast_values])
# plt.figure(figsize=(12, 6))
# plt.plot(full_data.index, full_data, color = 'blue', label = 'Dane historyczne')
# plt.plot(forecast_values, color='red', label = 'Wartości prognozowane')
# plt.axvline(x=unemployment_data.dropna().index[-1], linestyle='--', color='gray', label="Punkt startowy prognozy")
# plt.legend()
# plt.suptitle("Historyczny i prognozowany przebieg wartości - dane dot. stopy bezrobocia")
# plt.title(f"Metoda ARIMA")
# plt.grid()
# plt.show()

# predicted_unempl = unempl_model_fit.predict()
# AIC_unempl= round(unempl_model_fit.aic,2)
# mape_unempl = round(np.mean(np.abs((unemployment_data.value - predicted_unempl) / unemployment_data.value) * 100),2)
#
# plt.plot(unemployment_data.index, unemployment_data.value, color='red', label='Dane rzeczywiste')
# plt.plot(predicted_unempl, color='blue', label = 'Predykcje modelu')
# plt.legend()
# plt.suptitle("Dopasowanie modelu do danych (Stopa bezrobocia)")
# plt.title(f"AIC: {AIC_unempl}, MAPE: {mape_unempl}%")
# plt.show()
