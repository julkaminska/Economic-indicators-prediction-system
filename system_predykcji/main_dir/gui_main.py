import os
import sys
import pandas as pd

from gui_helpers import algorithm_mapping, indicator_mapping, plot_default, plot_forecast, time_unit_mapping, \
    return_dataframe, generate_report, save_plot_img
from PredictionModelClass import PredictionModel
from streamlit_extras.stylable_container import stylable_container
import streamlit as st

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    h1 {
        color: #E7473C;
        font-size: 2.5rem;
    }
    h2 {
        color: #E7473C;
        font-size: 2rem;
    }
    h3 {
        color: #E7473C;
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

datafile = 'main_dir/processed_data/indicators.csv'
data = pd.read_csv(datafile, index_col=0, parse_dates=True)

with stylable_container(
        key="container_with_border",
        css_styles="""
           {
               border: 1px solid rgba(58, 0, 30, 1.2);
               background-color: #ffffff;
               border-radius: 1rem;
               padding: calc(1em - 1px)
           }
           """):
    indicators = ["PKB", "Inflacja", "Stopa bezrobocia"]
    algorithms = ['ARIMA', "Sieć neuronowa"]

    st.markdown(
        f"""
        <h1 style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
            Predykcja wskaźników ekonomicznych
        </h1>
        """,
        unsafe_allow_html=True
    )
    # col1, col2 = st.columns([3,2])

    col1, col_blank, col2 = st.columns([5, 0.5, 3])
    col1.markdown(f"""
         <h3>
        Wykres przebiegu wartości wskaźnika
        </h3>
        """, unsafe_allow_html=True)
    chart_placeholder = col1.empty()
    col1a, col1b, rest = col1.columns([1,1,2])

    col2.markdown(f"""
             <h3>
            Dostosuj:
            </h3>
            """, unsafe_allow_html=True)
    col2a, col2b = col2.columns([2,1])
    time_placeholder = col2a.empty()
    indicator = col2a.selectbox("Wskaźnik ekonomiczny", indicators)
    algorithm = col2a.selectbox("Wykorzystany algorytm predykcji", algorithms)
    historical = col2.checkbox(label = "Wyświetl historyczny przebieg wskaźnika", value = False)
    col2a1, col2a2, col2a3 = col2.columns(3)
    forecast_click = col2a2.button("Wyświetl prognozę", type="primary")

    algorithm = algorithm_mapping[algorithm]
    indicator = indicator_mapping[indicator]

    if indicator == 'gdp':
        time_unit = col2b.radio("Jednostka czasu", options=("lata", "kwartały"))
    else:
        time_unit = col2b.radio("Jednostka czasu", options=("miesiące", "lata", "kwartały"))
    if time_unit == "miesiące":
        max_value = 60
    elif time_unit == "lata":
        max_value = 5
    else:
        max_value = 20
    n_steps = time_placeholder.number_input("Okres predykcji", value=5, step=1, min_value=1, max_value=max_value)
    time_unit = time_unit_mapping[time_unit]

    dataframe_expander = col1.expander('Pokaż dane w tabeli')

    if forecast_click:
        model = PredictionModel(indicator, algorithm)
        forecast = model.forecast(n_steps, time_unit)
        df = return_dataframe(indicator, algorithm, forecast=True, n_steps=n_steps, time_unit=time_unit)
        fig = plot_forecast(model, forecast, n_steps, show_historical=historical)
        plot_image = save_plot_img(fig)
        parameters, metrics = model.format_params(model.model_summary())
        report = generate_report(model, df, n_steps, time_unit, plot_image, parameters, metrics)
        st.toast(f"Raport predykcji zapisano w lokalizacji:   {report}")
        dataframe_expander.dataframe(return_dataframe(indicator, algorithm, forecast=True, n_steps= n_steps, time_unit=time_unit), hide_index = True, column_config = {"Rok": st.column_config.NumberColumn(format="%f")}, use_container_width = True)
        with col2.expander("Pokaż parametry modelu"):
            st.markdown(f"""<h3> Parametry modelu:</h3>""", unsafe_allow_html=True)
            # text = model.format_params(model.model_summary())
            for line in parameters:
                line = line.replace("; ", "\n\n")
                st.write(line)
            st.markdown(f"""<h3> Metryki dopasowania modelu:</h3>""", unsafe_allow_html=True)
            for line in metrics:
                st.write(line)
    else:
        dataframe_expander.dataframe(return_dataframe(indicator, algorithm, forecast=False, n_steps= n_steps, time_unit=time_unit), hide_index = True, column_config = {"Rok": st.column_config.NumberColumn(format="%f")}, use_container_width = True)
        fig = plot_default(indicator)

    chart_placeholder.plotly_chart(fig)