import os
import re
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF


indicator_mapping = {
        'PKB': 'gdp',
        'Inflacja': 'inflation',
        'Stopa bezrobocia': 'unemployment'}
indicator_reverse_mapping = {
        'gdp': 'PKB [mln PLN]',
        'inflation': 'Inflacja - Wskaznik CPI [%]',
        'unemployment': 'Stopa bezrobocia [%]'}

algorithm_mapping = {
        'ARIMA': 'arima',
        'Sieć neuronowa': 'nn'
    }
time_unit_mapping = {
        'miesiące' : 'M',
        'lata' : 'Y',
        'kwartały' : 'Q'
}

def calcualte_steps(n_steps, indicator, time_unit):
    if time_unit == "Q" and indicator != "gdp":
        n_steps = n_steps * 3
    if time_unit == "Y" and indicator != "gdp":
        n_steps = n_steps * 12
    if time_unit == "Y" and indicator == "gdp":
        n_steps = n_steps * 4
    if time_unit == "M" and indicator == "gdp":
        n_steps = n_steps % 3
    return n_steps

def return_dataframe(indicator, algorithm, forecast = False, n_steps = 0, time_unit = 'M'):
    historical_data_file = f"main_dir/processed_data/indicators.csv"
    with open(historical_data_file, 'r') as f:
        historical_data = pd.read_csv(f, index_col=0
                                      )
        if indicator == 'gdp':
            historical_data = historical_data[['year', 'quarter', indicator]]
            new_cols = {'year' : 'Rok', 'quarter' : 'Kwartal', f'{indicator}' : f'{indicator_reverse_mapping[indicator]}'}
        else:
            historical_data = historical_data[['year', 'month', indicator]]
            new_cols = {'year' : 'Rok', 'month' : 'Miesiac', f'{indicator}' : f'{indicator_reverse_mapping[indicator]}'}
        historical_data = historical_data.dropna()
    if forecast:
        forecasted_data_file = f"main_dir/pickles/forecasts/{indicator}_{algorithm}_forecast.csv"
        with open(forecasted_data_file, 'r') as f:
            forecast_df = pd.read_csv(f, index_col=0)
            forecast_df.index = pd.to_datetime(forecast_df.index)
        if indicator == 'gdp':
            last_2years = historical_data.tail(8)
            forecast_df["year"] = forecast_df.index.year
            forecast_df["quarter"] = forecast_df.index.quarter
        else:
            last_2years = historical_data.tail(24)
            forecast_df["year"] = forecast_df.index.year
            forecast_df["month"] = forecast_df.index.month
        forecast_len = calcualte_steps(n_steps, indicator, time_unit)
        forecast_df = forecast_df.head(forecast_len)
        forecast_df = forecast_df.rename(columns = {'value' : f'{indicator}'})

        combined_df = pd.concat([last_2years, forecast_df], ignore_index=True)
        if indicator == 'gdp':
            new_cols = {'year':'Rok', 'quarter': 'Kwartal', f'{indicator}' : f'{indicator_reverse_mapping[indicator]}'}
        else:
            new_cols = {'year':'Rok', 'month': 'Miesiac', f'{indicator}' : f'{indicator_reverse_mapping[indicator]}'}
        combined_df = combined_df.rename(columns = new_cols)
        return combined_df
    else:
        historical_data = historical_data.rename(columns = new_cols)
        return historical_data

def plot_default(indicator):
    datafile = 'main_dir/processed_data/indicators.csv'
    df = pd.read_csv(datafile, index_col=0, parse_dates=True)
    description = {
        'gdp': ['Produkt krajowy brutto (PKB) w Polsce', 'PKB [mln zł]'],
        'inflation': ['Poziom inflacji w Polsce<br>Wskaźnik CPI przy podstawie analogicznego miesiąca poprzedniego roku [%]',
            'Wskaźnik CPI [%]'],
        'unemployment': ['Stopa bezrobocia rejestrowanego w Polsce', 'Stopa bezrobocia [%]']}

    plot_titles = description[indicator]
    plot_data = df[indicator].dropna()
    if indicator == 'gdp':
        plot_data = plot_data.rolling(window=4).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data, mode='lines', marker=dict(size=4), name = 'Wartość wskaźnika', showlegend=True))
    fig.update_layout(title={'text': plot_titles[0], 'x': 0.5, 'xanchor': 'center'}, yaxis_title=plot_titles[1], title_font={'color': '#31333F'},
                      plot_bgcolor='white', paper_bgcolor='white')
    fig.update_yaxes(mirror=True, showline=True, linewidth=1, linecolor='#31333F')
    fig.update_xaxes(mirror=True, showline=True, linewidth=1, linecolor='#31333F')
    return fig


def plot_forecast(prediction_model, forecast, n_steps, show_historical=False):
    plot_titles = prediction_model.plot_descriptions()
    fig = go.Figure()
    if prediction_model.indicator == 'gdp':
        first_forecast_index = prediction_model.indicator_data.index[-1] + pd.DateOffset(months=3)
    else:
        first_forecast_index = prediction_model.indicator_data.index[-1] + pd.DateOffset(months=1)
    if forecast is None:
        full_data = prediction_model.indicator_data
        plot_data = full_data
        if prediction_model.indicator == "gdp":
            plot_data = plot_data.rolling(window=4).mean()
    else:
        if os.path.exists(prediction_model.forecast_file):
            full_data = pd.concat([prediction_model.indicator_data, forecast])

        if show_historical == False:
            plot_data = full_data[full_data.index >= first_forecast_index - pd.DateOffset(years=3)]
        else: plot_data = full_data

    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data.value,
        name="Wartość wskaźnika",
        mode='lines',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name="Wartości prognozowane",
        marker=dict(symbol='square', size=10, color='#ede9d0', opacity=0.4),
        showlegend=True
    ))

    fig.update_layout(title={'text': plot_titles[0], 'x': 0.5, 'xanchor': 'center'}, title_font={'color': '#31333F'},
                      plot_bgcolor='white', paper_bgcolor='white', yaxis_title=plot_titles[1], )
    fig.update_xaxes(mirror=True, showline=True, linewidth=1, linecolor='#31333F')
    fig.update_yaxes(mirror=True, showline=True, linewidth=1, linecolor='#31333F')
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=first_forecast_index, y0=0,
        x1=plot_data.index[-1], y1=1,
        fillcolor="#ede9d0", opacity=0.4, layer="below", line_width=0,
    )
    return fig

def generate_report(model, forecast, forecast_steps, time_unit, plot_img, parameters, metrics):
    output_file = f"main_dir/report/{model.indicator}_{model.method}_report.pdf"
    indicator_mapping = {
        'gdp': 'Produkt krajowy brutto (PKB) [mln PLN]',
        'inflation': 'Poziom inflacji (CPI - wskaznik cen towarow i uslug konsumpcyjnych) [%]',
        'unemployment': 'Stopa bezrobocia'}
    algorithm_mapping = {
        'arima': 'ARIMA',
        'nn': 'Siec neuronowa'
    }
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=" RAPORT: Prognoza wartosci wskaznikow ekonomicznych", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Wskaznik ekonomiczny: {indicator_mapping[model.indicator]}", ln=True)
    pdf.cell(200, 10, txt=f"Zakres danych historycznych: od {model.indicator_data.index[0].date()} do {model.indicator_data.index[-1].date()}", ln=True)
    pdf.cell(200, 10, txt=f"Okres prognozy: {forecast_steps} {time_unit}", ln=True)
    pdf.cell(200, 10, txt=f"Algorytm predykcji: {algorithm_mapping[model.method]}", ln=True)
    # parameters, metrics = model.format_params(model.model_summary())
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Parametry modelu:", ln=True)
    pdf.set_font("Arial", size=12)
    for line in parameters:
        # pdf.cell(200, 10, txt=line, ln=True)
        if len(str(line)) > 100:
            text_to_add = f"{str(line)}"
            pdf.multi_cell(110, 10, text_to_add)
        else:
            text_to_add = f"{str(line)}"
            pdf.cell(200, 10, txt=text_to_add, ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Metryki dopasowania modelu:", ln=True)
    pdf.set_font("Arial", size=12)
    for line in metrics:
        pdf.cell(200, 10, txt=line, ln=True)

    pdf.image(plot_img, x=10, y=pdf.get_y() + 10, w=180)

    pdf.add_page()

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Prognozowane wartosci:", ln=True)

    pdf.set_font("Arial", size = 8)
    n_cols = len(forecast.columns)
    col_width = 180 / n_cols
    headers = forecast.columns.tolist()
    for header in headers:
        pdf.cell(col_width, 10, header, border=1, align="C")
    pdf.ln()

    for row in forecast.itertuples(index=False):
        for value in row:
            pdf.cell(col_width, 10, str(value), border=1, align="C")
        pdf.ln()

    pdf.output(output_file)
    return output_file

def save_plot_img(fig):
    fig.write_image("main_dir/report/plot.png")
    return "main_dir/report/plot.png"
