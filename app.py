import streamlit as st

from pathlib import Path
import pandas as pd
import streamlit as st

import plotly.express as px

from pathlib import Path
import pandas as pd

import plotly.express as px

from utils.risk_dimensions import (
    calculate_risk_dimensions
)

from utils.comparison import (
    plot_risk_radar,
    plot_risk_evolution,
    plot_comparison_map,
    create_comparison_table,
    style_comparison_table,
    get_comparison_cards,
    create_quick_reading
)

from utils.clustering import run_cluster_analysis

from utils.prediction import (
    load_prediction_data,
    predict_country,
    score_to_rating,
    get_risk_level,
    get_agency_ratings,
    get_prediction_interpretation,
    plot_prediction_evolution,
    get_model_importance,
    get_available_prediction_years,
    get_available_countries,
    get_similar_countries,
    get_country_risk_profile
)

from utils.data_indicators import (
    get_data_countries,
    get_data_years,
    get_key_indicators,
    get_indicator_table,
    get_indicator_options,
    plot_indicator_evolution,
    get_country_download_data
)

from utils.prediction_report import (
    create_prediction_report
)

from utils.alerts import (
    get_country_alerts
)

from utils.full_report import (
    create_full_country_report
)

st.set_page_config(
    page_title="Análisis de Riesgo País",
    page_icon="🌍",
    layout="wide"
)

# Cargar ratings soberanos

DATA_PATH = Path("data/processed")

RATINGS_FILE = (
    DATA_PATH
    / "government_credit_rating_mean_2000_2025.csv"
)

ratings = pd.read_csv(
    RATINGS_FILE
)

ratings["risk_score"] = (
    100
    - ratings["rating_score_mean"]
)

@st.cache_data
def load_cluster_data():

    data = pd.read_csv(
        Path("data/model") /
        "country_risk_panel_integrated_raw_2000_2024.csv"
    )

    ratings = pd.read_csv(
        Path("data/processed") /
        "government_credit_rating_mean_2000_2025.csv"
    )

    return data, ratings


cluster_data, cluster_ratings = load_cluster_data()

# Cargar listado completo de economías

COUNTRIES_FILE = (
    DATA_PATH
    / "wdi_macro_panel_2000_2024.csv"
)

countries_panel = pd.read_csv(
    COUNTRIES_FILE
)

economies = (
    countries_panel[
        ["iso3", "country"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

total_economies = len(economies)


# Cargar base integrada de indicadores

MODEL_DATA_PATH = Path("data/model")

INDICATORS_FILE = (
    MODEL_DATA_PATH
    / "country_risk_panel_integrated_raw_2000_2024.csv"
)

indicators = pd.read_csv(
    INDICATORS_FILE
)


@st.cache_resource
def load_prediction_resources():

    return load_prediction_data()


prediction_data = load_prediction_resources()

prediction_model = prediction_data["model"]
prediction_preprocessor = prediction_data["preprocessor"]
prediction_features = prediction_data["features"]
prediction_feature_names = prediction_data["feature_names"]
prediction_panel = prediction_data["panel"]
prediction_agencies = prediction_data["agency_ratings"]

# Último año disponible

latest_rating_year = ratings["year"].max()

# Cobertura de ratings del último año

latest_year_ratings = ratings[
    ratings["year"] == latest_rating_year
][
    [
        "iso3",
        "rating_score_mean"
    ]
].copy()


# Añadir las 217 economías

latest_year_ratings = economies.merge(
    latest_year_ratings,
    on="iso3",
    how="left"
)


# Economías con rating

rated_latest = latest_year_ratings[
    latest_year_ratings["rating_score_mean"].notna()
].copy()


# Economías sin rating

unrated_latest = latest_year_ratings[
    latest_year_ratings["rating_score_mean"].isna()
].copy()


rated_count = len(rated_latest)

unrated_count = len(unrated_latest)


unrated_names = sorted(
    unrated_latest["country"]
    .dropna()
    .tolist()
)


unrated_text = ", ".join(
    unrated_names
)


economies_help = (
    f"Total de economías analizadas: {total_economies}. "
    f"En {latest_rating_year}, {rated_count} tienen rating "
    f"y {unrated_count} no disponen de rating.\n\n"
    f"Economías sin rating: {unrated_text}"
)

# Ratings disponibles en el último año

latest_ratings = latest_year_ratings[
    latest_year_ratings["rating_score_mean"].notna()
].copy()


# Rating medio

average_rating = (
    latest_ratings["rating_score_mean"]
    .mean()
)


# Risk Score

latest_ratings["risk_score"] = (
    100 - latest_ratings["rating_score_mean"]
)


average_risk_score = (
    latest_ratings["risk_score"]
    .mean()
)

# Número de economías con rating

rated_countries = (
    latest_ratings["iso3"]
    .nunique()
)

# Crear Risk Score

latest_ratings["risk_score"] = (
    100 - latest_ratings["rating_score_mean"]
)

# Año anterior

previous_rating_year = latest_rating_year - 1


# Ratings del año anterior

previous_ratings = ratings[
    ratings["year"] == previous_rating_year
].copy()


# Quedarnos con las columnas necesarias

current_compare = latest_ratings[
    [
        "iso3",
        "country",
        "rating_score_mean",
        "risk_score"
    ]
].copy()

previous_compare = previous_ratings[
    ["iso3", "rating_score_mean"]
].copy()

previous_compare = previous_compare.rename(
    columns={
        "rating_score_mean": "rating_score_mean_previous"
    }
)


# Unir año actual con año anterior

rating_comparison = current_compare.merge(
    previous_compare,
    on="iso3",
    how="left"
)


# Cambio del rating

rating_comparison["rating_change"] = (
    rating_comparison["rating_score_mean"]
    - rating_comparison["rating_score_mean_previous"]
)


# Clasificación del cambio

comparison_available = rating_comparison[
    rating_comparison["rating_change"].notna()
].copy()

improved_count = (
    comparison_available["rating_change"] > 0
).sum()

worsened_count = (
    comparison_available["rating_change"] < 0
).sum()

unchanged_count = (
    comparison_available["rating_change"] == 0
).sum()

# Países según el cambio de rating
improved_countries = (
    comparison_available[
        comparison_available["rating_change"] > 0
    ]
    [
        [
            "country",
            "rating_score_mean_previous",
            "rating_score_mean",
            "rating_change"
        ]
    ]
    .sort_values(
        "rating_change",
        ascending=False
    )
)

worsened_countries = (
    comparison_available[
        comparison_available["rating_change"] < 0
    ]
    [
        [
            "country",
            "rating_score_mean_previous",
            "rating_score_mean",
            "rating_change"
        ]
    ]
    .sort_values(
        "rating_change"
    )
)

unchanged_countries = (
    comparison_available[
        comparison_available["rating_change"] == 0
    ]
    [
        [
            "country",
            "rating_score_mean_previous",
            "rating_score_mean"
        ]
    ]
    .sort_values("country")
)


# Grupos de riesgo

risk_labels = [
    "Muy bajo",
    "Bajo",
    "Medio",
    "Alto",
    "Muy alto"
]

latest_ratings["risk_group"] = pd.cut(
    latest_ratings["risk_score"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=risk_labels,
    include_lowest=True
)


risk_group_counts = (
    latest_ratings["risk_group"]
    .value_counts()
    .reindex(risk_labels, fill_value=0)
)


# Países sin datos de rating en el último año

all_latest_year = ratings[
    ratings["year"] == latest_rating_year
].copy()

missing_latest_count = (
    all_latest_year["rating_score_mean"]
    .isna()
    .sum()
)

# Clasificar el nivel de riesgo

latest_ratings["risk_level"] = pd.cut(
    latest_ratings["risk_score"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=[
        "Muy bajo",
        "Bajo",
        "Medio",
        "Alto",
        "Muy alto"
    ],
    include_lowest=True
)

# Evolución histórica del Risk Score medio

risk_evolution = (
    ratings[
        ratings["rating_score_mean"].notna()
    ]
    .groupby("year", as_index=False)["rating_score_mean"]
    .mean()
)

risk_evolution["risk_score"] = (
    100 - risk_evolution["rating_score_mean"]
)


# Último año disponible de indicadores

latest_indicator_year = indicators["year"].max()
previous_indicator_year = latest_indicator_year - 1


current_indicators = indicators[
    indicators["year"] == latest_indicator_year
]

previous_indicators = indicators[
    indicators["year"] == previous_indicator_year
]


# Indicadores que aparecerán en Inicio

key_indicators = {
    "PIB real": "gdp_growth_pct",
    "Inflación": "inflation_cpi_pct",
    "Deuda pública": "government_debt_pct_gdp",
    "Cuenta corriente": "current_account_balance_pct_gdp",
    "Gobernanza": "government_effectiveness_estimate"
}


indicator_results = {}


for name, variable in key_indicators.items():

    current_value = (
        current_indicators[variable]
        .mean()
    )

    previous_value = (
        previous_indicators[variable]
        .mean()
    )

    indicator_results[name] = {
        "value": current_value,
        "change": current_value - previous_value
    }

# Estilo general  

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F3F8FD; 
    }

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6EAF0;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border-radius: 14px;
    }

    .app-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #6B7280;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .st-key-comparison_table_card,
    .st-key-comparison_map_card {
        background-color:#FFFFFF !important;
        border-radius:14px !important;
    }

    .st-key-comparison_table_card table {
        background-color:#FFFFFF !important;
        width:100% !important;
    }

    .st-key-comparison_table_card td,
    .st-key-comparison_table_card th {
        background-color:#FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

    /* Fondo blanco de los bloques de comparación */

    .st-key-comparison_table_card,
    .st-key-comparison_map_card,
    .st-key-comparison_radar_card,
    .st-key-comparison_evolution_card {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
    }

    .st-key-comparison_table_card
    [data-testid="stVerticalBlockBorderWrapper"],

    .st-key-comparison_map_card
    [data-testid="stVerticalBlockBorderWrapper"],

    .st-key-comparison_radar_card
    [data-testid="stVerticalBlockBorderWrapper"],

    .st-key-comparison_evolution_card
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# Estilo de las tarjetas de cambios

st.markdown(
    """
    <style>

    .st-key-improvement_card {
        background-color: #EAF8EF !important;
        border: 1px solid #B9E5C8 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        min-height: 180px;
    }

    .st-key-worsened_card {
        background-color: #FDEDED !important;
        border: 1px solid #F4C2C2 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        min-height: 180px;
    }

    .st-key-unchanged_card {
        background-color: #F2F4F7 !important;
        border: 1px solid #D8DDE5 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        min-height: 180px;
    }


    .st-key-improved_number button,
    .st-key-worsened_number button,
    .st-key-unchanged_number button {
        width: 100% !important;
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 0 !important;
    }


    .st-key-improved_number button p {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #1FA551 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    .st-key-worsened_number button p {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #E63946 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    .st-key-unchanged_number button p {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #596273 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }


    .st-key-improved_number button svg,
    .st-key-worsened_number button svg,
    .st-key-unchanged_number button svg {
        display: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

    /* Tarjetas */
    .st-key-comparison_cards
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: 12px;
    }

    /* Tabla */
    .st-key-comparison_table_map table {
        width: 100% !important;
    }


    /* Pantallas medianas */
    @media (max-width: 1100px) {

        /* Filtros */
        .st-key-comparison_filters
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        .st-key-comparison_filters
        [data-testid="column"] {
            flex: 1 1 220px !important;
            min-width: 220px !important;
        }


        /* Tarjetas */
        .st-key-comparison_cards
        [data-testid="column"] {
            flex: 1 1 30% !important;
            min-width: 210px !important;
        }


        /* Tabla + mapa */
        .st-key-comparison_table_map
        > div
        > [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        .st-key-comparison_table_map
        > div
        > [data-testid="stHorizontalBlock"]
        > [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }


    /* Pantallas pequeñas */
    @media (max-width: 700px) {

        .st-key-comparison_filters
        [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        .st-key-comparison_cards
        [data-testid="column"] {
            flex: 1 1 45% !important;
            min-width: 180px !important;
        }
    }


    /* Móvil */
    @media (max-width: 480px) {

        .st-key-comparison_cards
        [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>

    .st-key-cluster_pca_card,
    .st-key-cluster_size_card {
        background:#FFFFFF;
        border:1px solid #D7E3F0;
        border-radius:16px;
        padding:18px 22px 4px 22px;
        overflow:hidden;
    }

    .st-key-cluster_pca_card .stPlotlyChart,
    .st-key-cluster_size_card .stPlotlyChart {
        border-radius:16px;
        overflow:hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#-------------------------------------------------------------------------------------------
# Menú lateral
#-------------------------------------------------------------------------------------------

st.sidebar.markdown("### 🌍 Análisis de Riesgo País")

def go_to_prediction():
    st.session_state["navigation"] = "📈 Predicción"


page = st.sidebar.radio(
    "",
    [
        "🏠 Inicio",
        "👥 Comparación",
        "◉ Clústeres",
        "📈 Predicción",
        "📊 Datos e indicadores",
        "🔔 Alertas",
        "📄 Informes",
        "⚙️ Metodología"
    ],
    key="navigation"
)

#-------------------------------------------------------------------------------------------
# Página de inicio
#-------------------------------------------------------------------------------------------

if page == "🏠 Inicio":

    st.title("Análisis de Riesgo País")

    st.markdown(
        """
        <div class="app-subtitle">
            Análisis y predicción del riesgo soberano mediante
            indicadores económicos, fiscales, externos,
            institucionales y políticos.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            color:#8A94A3;
            font-size:12px;
            margin-top:-12px;
            margin-bottom:22px;
        ">
            Última actualización · Ratings hasta {latest_rating_year}
            · Indicadores hasta {latest_indicator_year}
        </div>
        """,
        unsafe_allow_html=True
    )


    # Tarjetas superiores ------------------------------------------------------------------
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:

        st.metric(
            label="🌍 Economías analizadas",
            value=total_economies,
            help=economies_help,
            border=True
        )
    
    
    with col2:
    
        st.metric(
            label="📅 Periodo",
            value=f"2000–{latest_rating_year}",
            help=(
                "Los indicadores explicativos abarcan 2000-2024 "
                "y los ratings soberanos están disponibles hasta 2025."
            ),
            border=True
        )
    
    
    with col3:
    
        st.metric(
            label=f"📊 Rating medio · {latest_rating_year}",
            value=f"{average_rating:.1f}",
            help=(
                "Rating soberano medio de las economías "
                "con calificación disponible. "
                "100 representa la mejor calidad crediticia."
            ),
            border=True
        )
    
    
    with col4:
    
        st.metric(
            label=f"⚠️ Risk Score medio · {latest_rating_year}",
            value=f"{average_risk_score:.1f}",
            help=(
                "Risk Score = 100 - Rating Score. "
                "Valores elevados representan mayor riesgo."
            ),
            border=True
        )
    
    # Boton directo a predicciones --------------------------------------------------------
    left_button, center_button, right_button = st.columns(
        [1, 1, 1]
    )
    
    with center_button:
    
        st.button(
            "📈 Realizar una predicción",
            type="primary",
            on_click=go_to_prediction,
            use_container_width=True
        )
    
    # Espacio reservado para el mapa -------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            f"#### 🌐 Mapa mundial de Risk Score · {latest_rating_year}"
        )
    
        fig_map = px.choropleth(
            latest_ratings,
            locations="iso3",
            locationmode="ISO-3",
            color="risk_score",
            hover_name="country",
            hover_data={
                "rating_score_mean": ":.1f",
                "risk_score": ":.1f",
                "iso3": False
            },
            color_continuous_scale=[
                [0.00, "#4F8EDC"],
                [0.20, "#5CC8A1"],
                [0.40, "#F4D35E"],
                [0.60, "#F6A04D"],
                [0.80, "#F36B4F"],
                [1.00, "#E63946"]
            ],
            range_color=(0, 100)
        )
    
    
        fig_map.update_geos(
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor="#ECEFF3",
            bgcolor="white",
            projection_type="natural earth"
        )
    
    
        fig_map.update_layout(
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=0
            ),
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            coloraxis_colorbar=dict(
                title="Risk Score",
                tickvals=[
                    10,
                    30,
                    50,
                    70,
                    90
                ],
                ticktext=[
                    "Muy bajo",
                    "Bajo",
                    "Medio",
                    "Alto",
                    "Muy alto"
                ]
            )
        )
    
    
        st.plotly_chart(
            fig_map,
            width="stretch"
        )

    # Resumen del riesgo actual ------------------------------------------------------------   
    st.markdown(
        '<div class="section-title">Resumen del riesgo actual</div>',
        unsafe_allow_html=True
    )
    
    col_change1, col_change2, col_change3 = st.columns(3)
    
    
    # Mejora
    
    with col_change1:
    
        with st.container(
            key="improvement_card"
        ):
    
            st.html(
                """
                <div style="
                    font-size:17px;
                    font-weight:700;
                    color:#23864B;
                ">
                    ↗ Mejora
                </div>
                """
            )
    
            with st.popover(
                str(improved_count),
                type="tertiary",
                width="stretch",
                key="improved_number",
                help="Pulsa para ver los países que han mejorado"
            ):
    
                st.markdown(
                    f"#### Países que mejoran · {latest_rating_year}"
                )
    
                improved_table = improved_countries.rename(
                    columns={
                        "country": "País",
                        "rating_score_mean_previous": f"Rating {previous_rating_year}",
                        "rating_score_mean": f"Rating {latest_rating_year}",
                        "rating_change": "Cambio"
                    }
                )
    
                st.dataframe(
                    improved_table,
                    hide_index=True,
                    width="stretch",
                    height=350
                )
    
            st.html(
                f"""
                <div style="
                    color:#64748B;
                    font-size:13px;
                    line-height:1.5;
                ">
                    Países que mejoran respecto a {previous_rating_year}
                </div>
                """
            )
    
    
    # Empeora
    
    with col_change2:
    
        with st.container(
            key="worsened_card"
        ):
    
            st.html(
                """
                <div style="
                    font-size:17px;
                    font-weight:700;
                    color:#C53A3A;
                ">
                    ↘ Empeora
                </div>
                """
            )
    
            with st.popover(
                str(worsened_count),
                type="tertiary",
                width="stretch",
                key="worsened_number",
                help="Pulsa para ver los países que han empeorado"
            ):
    
                st.markdown(
                    f"#### Países que empeoran · {latest_rating_year}"
                )
    
                worsened_table = worsened_countries.rename(
                    columns={
                        "country": "País",
                        "rating_score_mean_previous": f"Rating {previous_rating_year}",
                        "rating_score_mean": f"Rating {latest_rating_year}",
                        "rating_change": "Cambio"
                    }
                )
    
                st.dataframe(
                    worsened_table,
                    hide_index=True,
                    width="stretch",
                    height=350
                )
    
            st.html(
                f"""
                <div style="
                    color:#64748B;
                    font-size:13px;
                    line-height:1.5;
                ">
                    Países que empeoran respecto a {previous_rating_year}
                </div>
                """
            )
    
    
    # Sin cambios
    
    with col_change3:
    
        with st.container(
            key="unchanged_card"
        ):
    
            st.html(
                """
                <div style="
                    font-size:17px;
                    font-weight:700;
                    color:#596273;
                ">
                    → Sin cambios
                </div>
                """
            )
    
            with st.popover(
                str(unchanged_count),
                type="tertiary",
                width="stretch",
                key="unchanged_number",
                help="Pulsa para ver los países sin cambios"
            ):
    
                st.markdown(
                    f"#### Países sin cambios · {latest_rating_year}"
                )
    
                unchanged_table = unchanged_countries.rename(
                    columns={
                        "country": "País",
                        "rating_score_mean_previous": f"Rating {previous_rating_year}",
                        "rating_score_mean": f"Rating {latest_rating_year}"
                    }
                )
    
                st.dataframe(
                    unchanged_table,
                    hide_index=True,
                    width="stretch",
                    height=350
                )
    
            st.html(
                f"""
                <div style="
                    color:#64748B;
                    font-size:13px;
                    line-height:1.5;
                ">
                    Países que mantienen su rating respecto a {previous_rating_year}
                </div>
                """
            )   


    # Distribución del riesgo --------------------------------------------------------------
    st.markdown(
        '<div class="section-title">Distribución del riesgo actual</div>',
        unsafe_allow_html=True
    )

    st.markdown( #FFFFFF !important
    """
    <style>
    .st-key-risk_distribution {
        background-color: #E6EBF5;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    with st.container(border=True,
    key="risk_distribution"):
    
        cols = st.columns(6)
    
        risk_data = [
            ("#258BD2", "Muy bajo", "0 – 20", risk_group_counts["Muy bajo"]),
            ("#20C76F", "Bajo", "20 – 40", risk_group_counts["Bajo"]),
            ("#FFD21C", "Medio", "40 – 60", risk_group_counts["Medio"]),
            ("#FF7417", "Alto", "60 – 80", risk_group_counts["Alto"]),
            ("#ED1C2F", "Muy alto", "80 – 100", risk_group_counts["Muy alto"]),
            ("#D5D9DF", "Sin datos", "—", missing_latest_count)
        ]
    
        for col, (color, name, interval, count) in zip(cols, risk_data):
    
            with col:

                st.markdown(
                    f"""<div style="text-align:center;">
<div style="
width:20px;
height:20px;
border-radius:50%;
background-color:{color};
margin:0 auto 10px auto;
"></div>
<div style="font-weight:700;font-size:16px;">{name}</div>
<div style="font-size:12px;color:#8A94A3;margin-top:5px;">{interval}</div>
<div style="font-size:30px;font-weight:700;margin-top:20px;">{int(count)}</div>
</div>""",
                unsafe_allow_html=True
            )
    

    # Evolución e indicadores clave---------------------------------------------------------

    left, right = st.columns(
        [1.5, 1]
    )
    
    with left:
    
        with st.container(
            border=True,
            key="risk_evolution"
        ):
    
            st.markdown(
                "#### Evolución del Risk Score medio"
            )
    
            fig_risk = px.line(
                risk_evolution,
                x="year",
                y="risk_score",
                markers=True
            )
    
            fig_risk.update_traces(
                line=dict(
                    color="#258BD2",
                    width=3
                ),
                marker=dict(
                    size=6
                )
            )
    
            fig_risk.update_layout(
                height=370,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title="Risk Score",
                yaxis=dict(
                    range=[0, 100]
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                hovermode="x unified"
            )
    
            fig_risk.update_xaxes(
                showgrid=False
            )
    
            fig_risk.update_yaxes(
                gridcolor="#E8EDF3"
            )
    
            st.plotly_chart(
                fig_risk,
                width="stretch"
            )
    
    
    with right:
    
        with st.container(
            border=True,
            key="key_indicators"
        ):
        
            st.markdown(
                f"#### Indicadores clave · {latest_indicator_year}"
            )
        
        
            def show_indicator(name, value, change, decimals=1):

                if change > 0:
                    arrow = "↑"
                    color = "#1FA551"
                    background = "#EAF8EF"
            
                elif change < 0:
                    arrow = "↓"
                    color = "#E63946"  
                    background = "#FDEDED"
            
                else:
                    arrow = "→"
                    color = "#64748B"
                    background = "#F2F4F7"
            
            
                html = f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:10px 4px;
                    border-bottom:1px solid #EEF1F5;
                ">
                    <div>
                        <div style="
                            font-size:13px;
                            color:#475569;
                        ">
                            {name}
                        </div>
            
                        <div style="
                            font-size:25px;
                            font-weight:600;
                        ">
                            {value:.{decimals}f}
                        </div>
                    </div>
            
                    <div style="
                        background-color:{background};
                        color:{color};
                        border-radius:20px;
                        padding:5px 10px;
                        font-size:12px;
                        font-weight:600;
                        white-space:nowrap;
                    ">
                        {arrow} {abs(change):.{decimals}f} vs {previous_indicator_year}
                    </div>
                </div>
                """
            
                st.html(html)
                
            show_indicator(
                "PIB real (%)",
                indicator_results["PIB real"]["value"],
                indicator_results["PIB real"]["change"]
            )
    
            show_indicator(
                "Inflación (%)",
                indicator_results["Inflación"]["value"],
                indicator_results["Inflación"]["change"]
            )       
            show_indicator(
                "Deuda pública (% PIB)",
                indicator_results["Deuda pública"]["value"],
                indicator_results["Deuda pública"]["change"]
            )
            show_indicator(
                "Cuenta corriente (% PIB)",
                indicator_results["Cuenta corriente"]["value"],
                indicator_results["Cuenta corriente"]["change"]
            )
            show_indicator(
                "Gobernanza",
                indicator_results["Gobernanza"]["value"],
                indicator_results["Gobernanza"]["change"],
                decimals=2
            )
            
    st.markdown(
        """
        <style>
    
        .st-key-risk_evolution {
            background-color: #FFFFFF !important;
            border-radius: 14px;
        }
    
        .st-key-key_indicators {
            background-color: #FFFFFF !important;
            border-radius: 14px;
        }
    
        </style>
        """,
        unsafe_allow_html=True
    )

   


    # Firma ------------------------------------------------------------------
    st.markdown(
        """<div style="text-align:center; margin-top:60px; padding-top:20px; color:#A0A6B0; font-size:12px; line-height:1.7;">
        <div style="font-family:monospace; letter-spacing:3px; font-size:11px;">m o-o l o-a l o</div>
        <div style="font-size:11px;">TFM · Máster en Big Data, Data Science e Inteligencia Artificial</div>
        <div style="font-size:13px;">Mónica López Alonso</div>
        <div style="font-size:11px;">molopezalonso@gmail.com</div>
        </div>""",
        unsafe_allow_html=True
    )


#-------------------------------------------------------------------------------------------
# Página de Comparación
#-------------------------------------------------------------------------------------------

elif page == "👥 Comparación":

    st.title("Comparación de países")

    st.markdown(
        """
        <div class="app-subtitle">
            Compara el perfil de riesgo, los principales indicadores
            y la evolución histórica de dos economías.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Años disponibles
    comparison_years = sorted(
        indicators["year"]
        .dropna()
        .unique(),
        reverse=True
    )
    
   # Tres selectores en una misma fila
    with st.container(key="comparison_filters"):
    
        col_year, col_country_a, col_country_b = st.columns(
            [0.7, 1.15, 1.15]
        )
    
        # Año
        with col_year:
    
            selected_year = st.selectbox(
                "📅 Año",
                comparison_years,
                index=0
            )
    
        # Países disponibles para el año seleccionado
        countries_year = (
            indicators[
                indicators["year"] == selected_year
            ][
                ["iso3", "country"]
            ]
            .dropna()
            .drop_duplicates()
        )
    
        ratings_year_available = (
            ratings[
                (ratings["year"] == selected_year)
                & ratings["rating_score_mean"].notna()
            ][
                ["iso3"]
            ]
            .drop_duplicates()
        )
    
        available_countries = (
            countries_year
            .merge(
                ratings_year_available,
                on="iso3",
                how="inner"
            )
            .sort_values("country")
            .reset_index(drop=True)
        )
    
        country_name_to_iso = dict(
            zip(
                available_countries["country"],
                available_countries["iso3"]
            )
        )
    
        country_names = (
            available_countries["country"]
            .tolist()
        )
    
        # País A por defecto
        default_a = (
            country_names.index("Spain")
            if "Spain" in country_names
            else 0
        )
    
        # País A
        with col_country_a:
    
            country_a_name = st.selectbox(
                "🔵 País A",
                country_names,
                index=default_a,
                key="comparison_country_a"
            )
    
        # Evitar seleccionar el mismo país
        country_b_options = [
            country
            for country in country_names
            if country != country_a_name
        ]
    
        # País B por defecto
        default_b = (
            country_b_options.index("Italy")
            if "Italy" in country_b_options
            else 0
        )
    
        # País B
        with col_country_b:
    
            country_b_name = st.selectbox(
                "🟢 País B",
                country_b_options,
                index=default_b,
                key="comparison_country_b"
            )
    
    country_a = (
        country_name_to_iso[
            country_a_name
        ]
    )
    
    country_b = (
        country_name_to_iso[
            country_b_name
        ]
    )

    # Calcular dimensiones
    risk_dimensions = (
        calculate_risk_dimensions(
            indicators,
            selected_year
        )
    )

    # Datos de las tarjetas
    cards = get_comparison_cards(
        indicators,
        ratings,
        selected_year,
        country_a,
        country_b
    )

    # Tarjetas superiores
    metrics = [
        ("⚠️", "Risk Score", 1),
        ("📊", "Rating", 1),
        ("📈", "PIB real (%)", 2),
        ("🔥", "Inflación (%)", 2),
        ("🏦", "Deuda pública (% PIB)", 2)
    ]

    with st.container(key="comparison_cards"):

        card_columns = st.columns(5)
    
        for col, (
            icon,
            metric,
            decimals
        ) in zip(
            card_columns,
            metrics
        ):
    
            value_a = cards[country_a][metric]
            value_b = cards[country_b][metric]
    
            with col:
    
                st.html(
                    f"""
                    <div style="
                        background:#FFFFFF;
                        border:1px solid #D1D5DB;
                        border-radius:16px;
                        padding:24px 28px;
                        height:190px;
                        box-sizing:border-box;
                        display:flex;
                        flex-direction:column;
                    ">
    
                        <div style="
                            font-size:16px;
                            font-weight:700;
                            color:#1E293B;
                            margin-bottom:22px;
                        ">
                            {icon} {metric}
                        </div>
    
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            gap:18px;
                            margin-top:auto;
                        ">
    
                            <div style="
                                flex:1;
                            ">
    
                                <div style="
                                    color:#258BD2;
                                    font-size:13px;
                                    font-weight:600;
                                    margin-bottom:10px;
                                ">
                                    {country_a_name}
                                </div>
    
                                <div style="
                                    color:#1E293B;
                                    font-size:26px;
                                    font-weight:700;
                                    white-space:nowrap;
                                ">
                                    {value_a:.{decimals}f}
                                </div>
    
                            </div>
    
                            <div style="
                                flex:1;
                            ">
    
                                <div style="
                                    color:#20B26B;
                                    font-size:13px;
                                    font-weight:600;
                                    margin-bottom:10px;
                                ">
                                    {country_b_name}
                                </div>
    
                                <div style="
                                    color:#1E293B;
                                    font-size:26px;
                                    font-weight:700;
                                    white-space:nowrap;
                                ">
                                    {value_b:.{decimals}f}
                                </div>
    
                            </div>
    
                        </div>
    
                    </div>
                    """
                )

    # Tabla comparativa y mapa
    st.markdown(
        '<div class="section-title">Comparación de indicadores clave</div>',
        unsafe_allow_html=True
    )
    
    comparison_table = (
        create_comparison_table(
            indicators,
            selected_year,
            country_a,
            country_b
        )
    )
    
    styled_table = (
        style_comparison_table(
            comparison_table,
            country_a_name,
            country_b_name
        )
    )


    with st.container(key="comparison_table_map"):            
        table_col, map_col = st.columns(
            [1.15, 1]
        )
        
        with table_col:
        
            with st.container(
                border=True,
                key="comparison_table_card"
            ):
        
                st.html(
                    styled_table.to_html()
                )
        
        with map_col:
        
            with st.container(
                border=True,
                key="comparison_map_card"
            ):
        
                fig_comparison_map = (
                    plot_comparison_map(
                        indicators,
                        country_a,
                        country_b
                    )
                )
        
                st.plotly_chart(
                    fig_comparison_map,
                    width="stretch"
                )


    
    # Radar y evolución
    left_chart, right_chart = st.columns(2)

    with left_chart:

        with st.container(
            border=True,
            key="comparison_radar_card"
        ):
            st.markdown(
                """
                <div style="
                    font-size:18px;
                    font-weight:700;
                    color:#1E293B;
                    margin-bottom:2px;
                ">
                    Comparación dimensiones de riesgo
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_radar = plot_risk_radar(
                risk_dimensions,
                country_a,
                country_b
            )

            st.plotly_chart(
                fig_radar,
                width="stretch"
            )

            with st.popover(
                "ℹ️ ¿Cómo se calculan las dimensiones?"
            ):

                st.markdown(
                    """
                    Los indicadores se transforman anualmente
                    a una escala de riesgo de 0 a 100 mediante
                    percentiles.

                    **0** representa menor vulnerabilidad relativa
                    y **100** mayor vulnerabilidad relativa.

                    Se resumen seis dimensiones:

                    - Macroeconómica
                    - Fiscal
                    - Externa
                    - Institucional
                    - Política y estabilidad
                    - Historial crediticio
                    """
                )

    with right_chart:

        with st.container(
            border=True,
            key="comparison_evolution_card"
        ):

            fig_evolution = (
                plot_risk_evolution(
                    ratings,
                    indicators,
                    country_a,
                    country_b
                )
            )

            st.plotly_chart(
                fig_evolution,
                width="stretch"
            )

    # Lectura rápida
    quick_reading = (
        create_quick_reading(
            risk_dimensions,
            cards,
            country_a,
            country_b
        )
    )

    st.html(
        f"""
        <div style="
            background:#F0F9F4;
            border:1px solid #B7E4C7;
            border-radius:14px;
            padding:18px 22px;
            margin-top:12px;
        ">

            <div style="
                font-weight:700;
                color:#15803D;
                margin-bottom:6px;
            ">
                💡 Lectura rápida
            </div>

            <div style="
                color:#334155;
                font-size:14px;
                line-height:1.6;
            ">
                {quick_reading}
            </div>

        </div>
        """,
    )

    # Firma
    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:50px;
            color:#B0B7C3;
            font-family:monospace;
            letter-spacing:3px;
            font-size:11px;
        ">
            m o-o l o-a l o
        </div>
        """,
        unsafe_allow_html=True
    )




#-------------------------------------------------------------------------------------------
# Pagina de Clusteres
#-------------------------------------------------------------------------------------------

elif page == "◉ Clústeres":
    st.title("Análisis de clústeres")
    # paises
    EUROPE_ISO3 = [
        "ALB", "AND", "AUT", "BEL", "BGR",
        "BIH", "BLR", "CHE", "CYP", "CZE",
        "DEU", "DNK", "ESP", "EST", "FIN",
        "FRA", "GBR", "GRC", "HRV", "HUN",
        "IRL", "ISL", "ITA", "LIE", "LTU",
        "LUX", "LVA", "MDA", "MKD", "MLT",
        "MNE", "NLD", "NOR", "POL", "PRT",
        "ROU", "RUS", "SRB", "SVK", "SVN",
        "SWE", "UKR"
    ]
    
    AUSTRALIA_ISO3 = [
        "AUS"
    ]

    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster_years = sorted(
            cluster_data["year"]
            .dropna()
            .unique(),
            reverse=True
        )
        
        selected_year = st.selectbox(
            "📅 Año",
            cluster_years,
            index=0
        )
    
    with col2:
        region_options = [
            "Todas",
            "Europa",
            "América",
            "Europa + América",
            "Europa + América + Australia"
        ]
    
        selected_region = st.selectbox(
            "🌍 Región",
            region_options
        )
    
    with col3:
        cluster_option = st.selectbox(
            "🧩 Número de clústeres",
            [
                "Automático",
                3,
                4,
                5,
                6
            ]
        )
    selected_k = (
        None
        if cluster_option == "Automático"
        else cluster_option
    )
    
    cluster_data_selected = cluster_data.copy()
    region_filter = None
    
    # Países americanos
    america_mask = (
        cluster_data["region"]
        .str.contains(
            "America|Caribbean",
            case=False,
            na=False
        )
    )
    
    if selected_region == "Europa":
    
        cluster_data_selected = cluster_data[
            cluster_data["iso3"].isin(
                EUROPE_ISO3
            )
        ].copy()
    
    elif selected_region == "América":
    
        cluster_data_selected = cluster_data[
            america_mask
        ].copy()
    
    elif selected_region == "Europa + América":
    
        cluster_data_selected = cluster_data[
            cluster_data["iso3"].isin(
                EUROPE_ISO3
            )
            | america_mask
        ].copy()
    
    elif selected_region == "Europa + América + Australia":
    
        cluster_data_selected = cluster_data[
            cluster_data["iso3"].isin(
                EUROPE_ISO3
            )
            | america_mask
            | cluster_data["iso3"].isin(
                AUSTRALIA_ISO3
            )
        ].copy()

    cluster_results = run_cluster_analysis(
        data=cluster_data_selected,
        ratings=cluster_ratings,
        year=selected_year,
        region=region_filter,
        selected_k=selected_k
    )

    # Resumen del análisis
    c1, c2, c3, c4 = st.columns(4)
    
    cards_cluster = [
        (
            c1,
            "🌍",
            "Países",
            cluster_results["countries_final"],
            ""
        ),
        (
            c2,
            "📊",
            "Variables",
            len(cluster_results["variables"]),
            ""
        ),
        (
            c3,
            "📉",
            "Componentes PCA",
            cluster_results["n_components"],
            (
                f"Varianza explicada: "
                f"{cluster_results['pca_variance'] * 100:.1f}%"
            )
        ),
        (
            c4,
            "🧩",
            "Clústeres",
            cluster_results["best_k"],
            ""
        )
    ]

    for col, icon, title, value, subtitle in cards_cluster:

        with col:
    
            st.html(
                f"""
                <div style="
                    background:#EAF4FC;
                    border:1px solid #BDD8EB;
                    border-radius:16px;
                    padding:24px 28px;
                    height:170px;
                    box-sizing:border-box;
                    display:flex;
                    flex-direction:column;
                ">
    
                    <div style="
                        font-size:16px;
                        font-weight:700;
                        color:#1E293B;
                        margin-bottom:28px;
                    ">
                        {icon} {title}
                    </div>
    
                    <div style="
                        font-size:38px;
                        font-weight:700;
                        color:#1E293B;
                        line-height:1;
                        text-align:center;
                    ">
                        {value}
                    </div>
    
                    <div style="
                        font-size:12px;
                        font-weight:400;
                        color:#64748B;
                        margin-top:22px;
                        line-height:1.6;
                    ">
                        {subtitle}
                    </div>
    
                </div>
                """
            )

    #CLUSTER    
    st.markdown("### Distribución de los países")

    with st.container(
        key="cluster_pca_card"
    ):
    
        st.plotly_chart(
            cluster_results["fig_pca"],
            use_container_width=True
        )

    # tarjetitas
    summary = cluster_results["summary"]

    lowest_risk = summary.iloc[0]
    highest_risk = summary.iloc[-1]
    
    largest_cluster = summary.loc[
        summary["Países"].idxmax()
    ]

    info_col, findings_col, profile_col = st.columns(
        [0.9, 2.0, 1.1]
    )
    
    # Información PCA
    with info_col:

        st.html(
            f"""
            <div style="
                background:#F8FAFC;
                border:1px solid #DCE4ED;
                border-radius:10px;
                padding:12px 16px;
                height:230;
                box-sizing:border-box;
                font-size:12px;
                line-height:1.5;
                color:#64748B;
            ">
    
                <span style="
                    color:#258BD2;
                    font-weight:600;
                    margin-right:6px;
                ">
                    ⓘ
                </span>
    
                El gráfico muestra la proyección de los países
                en las dos primeras componentes principales (PCA).
    
                <div style="
                    margin-top:7px;
                    font-weight:600;
                    color:#475569;
                ">
                    Varianza explicada:
                    {cluster_results["pca_variance"] * 100:.1f}%
                </div>
    
            </div>
            """
        )
    
    
    # Hallazgos rápidos
    with findings_col:
    
        findings = cluster_results["findings"][:3]
    
        findings_html = ""
    
        for finding in findings:
    
            findings_html += f"""
            <div style="
                margin-bottom:8px;
                display:flex;
                gap:8px;
                align-items:flex-start;
            ">
                <span style="
                    color:#20B26B;
                    font-weight:700;
                ">
                    ✓
                </span>
    
                <span>
                    {finding}
                </span>
            </div>
            """
    
        st.html(
            f"""
            <div style="
                background:#F0F9F4;
                border:1px solid #B7E4C7;
                border-radius:10px;
                padding:12px 18px;
                height:230;
                box-sizing:border-box;
                font-size:12px;
                line-height:1.45;
                color:#475569;
            ">
    
                <div style="
                    font-size:13px;
                    font-weight:600;
                    color:#15803D;
                    margin-bottom:9px;
                ">
                    💡 Hallazgos rápidos
                </div>
    
                {findings_html}
    
            </div>
            """
        )

    with profile_col:

        st.html(
            f"""
            <div style="
                background:#F8FAFC;
                border:1px solid #DCE4ED;
                border-radius:10px;
                padding:12px 16px;
                height:230;
                box-sizing:border-box;
                font-size:12px;
                line-height:1.55;
                color:#475569;
            ">
    
                <div style="
                    font-size:13px;
                    font-weight:600;
                    color:#1E293B;
                    margin-bottom:12px;
                ">
                    📌 Resumen de perfiles
                </div>
    
                <div style="margin-bottom:8px;">
                    🔵 <b>Menor riesgo:</b><br>
                    Clúster {int(lowest_risk["Cluster"])}
                    · {lowest_risk["Risk Score medio"]:.1f}
                </div>
    
                <div style="margin-bottom:8px;">
                    🔴 <b>Mayor riesgo:</b><br>
                    Clúster {int(highest_risk["Cluster"])}
                    · {highest_risk["Risk Score medio"]:.1f}
                </div>
    
                <div>
                    👥 <b>Mayor grupo:</b><br>
                    Clúster {int(largest_cluster["Cluster"])}
                    · {int(largest_cluster["Países"])} países
                </div>
    
            </div>
            """
        )

    # TABLA 
    st.markdown("### Resumen de los clústeres")

    table_col, size_col = st.columns(
        [1.7, 1]
    )
    
    # Tabla
    with table_col:
    
        st.html(
            cluster_results["table_html"]
        )
    
    
    # Tamaño de los clústeres
    with size_col:
    
        fig_sizes = cluster_results["fig_sizes"]
    
        fig_sizes.update_layout(
            height=480,
            margin=dict(
                l=20,
                r=30,
                t=55,
                b=45
            )
        )
    
        with st.container(
            key="cluster_size_card"
        ):
        
            st.plotly_chart(
                cluster_results["fig_sizes"],
                use_container_width=True
            )

    # firma ------------------------
    st.markdown(
        """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
        m o-o l o-a l o
        </div>""",
        unsafe_allow_html=True
    )




#-------------------------------------------------------------------------------------------
# Pagina de Prediccion
#-------------------------------------------------------------------------------------------

elif page == "📈 Predicción":

    st.title(
        "Predicción del rating soberano"
    )

    st.markdown(
        """
        Estimación del rating soberano futuro mediante
        el modelo Random Forest definitivo.
        """
    )


    # Selección de país
    countries_df = get_available_countries(
        prediction_panel
    )

    country_options = dict(
        zip(
            countries_df["country"],
            countries_df["iso3"]
        )
    )


    col1, col2, col3 = st.columns(
        [1.2, 1, 0.7]
    )


    with col1:

        selected_country_name = st.selectbox(
            "🌍 País",
            list(
                country_options.keys()
            ),
            index=(
                list(
                    country_options.keys()
                ).index("Spain")
                if "Spain"
                in country_options
                else 0
            )
        )


    selected_country_iso = (
        country_options[
            selected_country_name
        ]
    )


    available_years = (
        get_available_prediction_years(
            prediction_panel,
            selected_country_iso
        )
    )


    with col2:

        selected_prediction_year = (
            st.selectbox(
                "📅 Año de referencia",
                available_years,
                index=0
            )
        )

        target_year = (
            selected_prediction_year
            + 1
        )
    with col3: 

        st.selectbox(
            "📈 Predicción",
            [target_year],
            index=0,
            key="prediction_year_display"
        )

    st.caption(
        f"Los indicadores de {selected_prediction_year} "
        f"se utilizan para estimar el rating soberano de {target_year}."
    )


    # Predicción
    prediction_result = predict_country(
        panel=prediction_panel,
        model=prediction_model,
        preprocessor=prediction_preprocessor,
        features=prediction_features,
        country_iso=selected_country_iso,
        year=selected_prediction_year
    )


    prediction = (
        prediction_result[
            "prediction"
        ]
    )

    current_rating = (
        prediction_result[
            "current_rating"
        ]
    )

    change = (
        prediction_result[
            "change"
        ]
    )

    target_year = (
        prediction_result[
            "target_year"
        ]
    )


    equivalent_rating = (
        score_to_rating(
            prediction
        )
    )

    risk_level = (
        get_risk_level(
            prediction
        )
    )


    # Color del nivel de riesgo
    if risk_level == "Muy bajo":

        risk_color = "#16A34A"
        risk_background = "#DCFCE7"

    elif risk_level == "Bajo":

        risk_color = "#15803D"
        risk_background = "#E8F5E9"

    elif risk_level == "Medio":

        risk_color = "#B45309"
        risk_background = "#FEF3C7"

    elif risk_level == "Alto":

        risk_color = "#EA580C"
        risk_background = "#FFEDD5"

    else:

        risk_color = "#DC2626"
        risk_background = "#FEE2E2"


    # Cambio
    if pd.isna(change):

        change_text = "."
        change_color = "#64748B"
        change_icon = ""

    elif change > 0:

        change_text = (
            f"+{change:.1f} puntos"
        )

        change_color = "#16A34A"
        change_icon = "↑"

    elif change < 0:

        change_text = (
            f"{change:.1f} puntos"
        )

        change_color = "#DC2626"
        change_icon = "↓"

    else:

        change_text = "0.0 puntos"
        change_color = "#64748B"
        change_icon = "→"
    
    fig_evolution = plot_prediction_evolution(
        panel=prediction_panel,
        country_iso=selected_country_iso,
        year=selected_prediction_year,
        prediction=prediction
    )
    
    prediction_col, evolution_col = st.columns(
        [1, 2.6],
        gap="large"
    )

    st.markdown(
        """
        <style>
    
        .st-key-evolution_prediction {
            background:#FFFFFF;
            border:1px solid #D7E3F0;
            border-radius:16px;
            padding:18px 22px 4px 22px;
            overflow:hidden;
            margin-top:24px;
        }
    
        .st-key-evolution_prediction .stPlotlyChart {
            border-radius:16px;
            overflow:hidden;
        }
    
        </style>
        """,
        unsafe_allow_html=True
    )

    with prediction_col:
    
        # Tarjeta principal
        st.html(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #D7E3F0;
                border-radius:16px;
                padding:28px 32px;
                max-width:520px;
                margin-top:20px;
                margin-bottom:20px;
                box-sizing:border-box;
            ">
    
                <div style="
                    font-size:18px;
                    color:#1E293B;
                    margin-bottom:28px;
                ">
                    Predicción del modelo
                </div>
    
                <div style="
                    text-align:center;
                    font-size:58px;
                    font-weight:700;
                    color:#1E293B;
                    line-height:1;
                ">
                    {prediction:.1f}
    
                    <span style="
                        font-size:24px;
                        font-weight:400;
                    ">
                        /100
                    </span>
                </div>
    
                <div style="
                    text-align:center;
                    margin-top:18px;
                    color:#64748B;
                    font-size:16px;
                ">
                    Rating soberano estimado para {target_year}
                </div>
    
                <div style="
                    background:{risk_background};
                    color:{risk_color};
                    border-radius:22px;
                    padding:8px 20px;
                    width:max-content;
                    margin:22px auto;
                    font-size:15px;
                ">
                    ● Riesgo {risk_level.lower()}
                </div>
    
                <div style="
                    text-align:center;
                    color:#64748B;
                    font-size:15px;
                    margin-top:25px;
                ">
                    Equivalencia aproximada agencias
                </div>
    
                <div style="
                    text-align:center;
                    font-size:36px;
                    font-weight:700;
                    color:#1E293B;
                    margin-top:5px;
                ">
                    {equivalent_rating}
                </div>
    
                <div style="
                    border-top:1px solid #E2E8F0;
                    margin-top:25px;
                    padding-top:25px;
                    text-align:center;
                ">
    
                    <div style="
                        color:#64748B;
                        font-size:15px;
                    ">
                        Cambio respecto al rating medio del año seleccionado
                    </div>
    
                    <div style="
                        font-size:29px;
                        font-weight:700;
                        color:{change_color};
                        margin-top:10px;
                    ">
                        {change_icon}
                        {change_text}
                    </div>
    
                </div>
    
            </div>
            """
        )
    with evolution_col:

        with st.container(
            key="evolution_prediction"
        ):
    
            st.plotly_chart(
                fig_evolution,
                use_container_width=True
            )


    # Ratings originales de las agencias
    agency_ratings = get_agency_ratings(
        prediction_agencies,
        selected_country_iso,
        selected_prediction_year
    )


    interpretation = (
        get_prediction_interpretation(
            prediction,
            current_rating
        )
    )


    agency_col, interpretation_col = (
        st.columns(2)
    )


    with agency_col:

        st.html(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #D7E3F0;
                border-radius:16px;
                padding:24px;
                height:220px;
                box-sizing:border-box;
            ">

                <div style="
                    font-size:17px;
                    color:#1E293B;
                    margin-bottom:30px;
                ">
                    🏢 Calificaciones de las agencias
                </div>

                <div style="
                    display:flex;
                    justify-content:space-around;
                    text-align:center;
                ">

                    <div>

                        <div style="
                            color:#64748B;
                            font-size:14px;
                        ">
                            S&P
                        </div>

                        <div style="
                            font-size:32px;
                            font-weight:700;
                            color:#1E293B;
                            margin-top:8px;
                        ">
                            {agency_ratings["S&P"]}
                        </div>

                    </div>

                    <div>

                        <div style="
                            color:#64748B;
                            font-size:14px;
                        ">
                            Moody's
                        </div>

                        <div style="
                            font-size:32px;
                            font-weight:700;
                            color:#1E293B;
                            margin-top:8px;
                        ">
                            {agency_ratings["Moody's"]}
                        </div>

                    </div>

                    <div>

                        <div style="
                            color:#64748B;
                            font-size:14px;
                        ">
                            Fitch
                        </div>

                        <div style="
                            font-size:32px;
                            font-weight:700;
                            color:#1E293B;
                            margin-top:8px;
                        ">
                            {agency_ratings["Fitch"]}
                        </div>

                    </div>

                </div>

            </div>
            """
        )


    with interpretation_col:

        st.html(
            f"""
            <div style="
                background:#F0F9F4;
                border:1px solid #B7E4C7;
                border-radius:16px;
                padding:24px;
                height:220px;
                box-sizing:border-box;
            ">

                <div style="
                    font-size:17px;
                    color:#1E293B;
                    margin-bottom:25px;
                ">
                    💡 Interpretación rápida
                </div>

                <div style="
                    font-size:16px;
                    font-weight:600;
                    color:#334155;
                    margin-bottom:12px;
                ">
                    {interpretation["icon"]}
                    {interpretation["status"]}
                </div>

                <div style="
                    font-size:14px;
                    line-height:1.55;
                    color:#64748B;
                ">
                    {interpretation["text"]}
                </div>

                <div style="
                    margin-top:15px;
                    font-size:14px;
                    color:#475569;
                ">
                    Variación estimada:
                    <b>{change_text}</b>
                </div>

            </div>
            """
        )


    # Información del modelo
    st.html(
        f"""
        <div style="
            background:#F8FAFC;
            border:1px solid #DCE4ED;
            border-radius:16px;
            padding:20px 24px;
            margin-top:18px;
            margin-bottom:30px;
            box-sizing:border-box;
        ">

            <div style="
                font-size:16px;
                color:#1E293B;
                margin-bottom:14px;
            ">
                📌 Sobre la predicción
            </div>

            <div style="
                display:flex;
                gap:45px;
                flex-wrap:wrap;
                font-size:14px;
                color:#64748B;
            ">

                <div>
                    Modelo
                    <br>
                    <b style="
                        color:#334155;
                    ">
                        Random Forest
                    </b>
                </div>

                <div>
                    Variables fundamentales
                    <br>
                    <b style="
                        color:#334155;
                    ">
                        {len(prediction_features)}
                    </b>
                </div>

                <div>
                    Horizonte
                    <br>
                    <b style="
                        color:#334155;
                    ">
                        Rating t+1
                    </b>
                </div>

                <div>
                    MAE test
                    <br>
                    <b style="
                        color:#334155;
                    ">
                        4.988 puntos
                    </b>
                </div>

                <div>
                    R² test
                    <br>
                    <b style="
                        color:#334155;
                    ">
                        0.928
                    </b>
                </div>

            </div>

        </div>
        """
    )

    # PERFILES SIMILAR4ES

    similar_cluster_results = run_cluster_analysis(
        data=cluster_data,
        ratings=cluster_ratings,
        year=selected_prediction_year,
        region=None,
        selected_k=None
    )
    
    
    similar_results = get_similar_countries(
        cluster_results=similar_cluster_results,
        country_iso=selected_country_iso,
        panel=prediction_panel,
        model=prediction_model,
        preprocessor=prediction_preprocessor,
        features=prediction_features,
        year=selected_prediction_year,
        top_n=3
    )

    country_risk_profile = get_country_risk_profile(
        data=cluster_data,
        country_iso=selected_country_iso,
        year=selected_prediction_year,
        top_n=2
    )

    similar_html = ""

    countries_similar = similar_results["countries"]
    
    for i, country in enumerate(countries_similar):
    
        border = (
            "none"
            if i == len(countries_similar) - 1
            else "1px solid #E2E8F0"
        )
    
        similar_html += f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:13px 0;
            border-bottom:{border};
        ">
    
            <div style="
                font-size:15px;
                color:#334155;
            ">
                🌍 {country["country"]}
            </div>
    
            <div style="
                text-align:right;
            ">
    
                <div style="
                    font-size:18px;
                    font-weight:700;
                    color:#1E293B;
                ">
                    {country["prediction"]:.1f}
                    <span style="
                        font-size:12px;
                        font-weight:400;
                        color:#64748B;
                    ">
                        /100
                    </span>
                </div>
    
                <div style="
                    font-size:12px;
                    color:#64748B;
                ">
                    {country["rating"]}
                </div>
    
            </div>
    
        </div>
        """
    
    strengths_html = ""
    
    for item in country_risk_profile["strengths"]:
    
        strengths_html += f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:10px 0;
            border-bottom:1px solid #E2E8F0;
        ">
    
            <div style="
                font-size:14px;
                color:#334155;
            ">
                ✓ {item["dimension"]}
            </div>
    
            <div style="
                font-size:15px;
                font-weight:600;
                color:#16A34A;
            ">
                {item["score"]:.1f}
            </div>
    
        </div>
        """
    
    
    weaknesses_html = ""
    
    for item in country_risk_profile["weaknesses"]:
    
        weaknesses_html += f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:10px 0;
            border-bottom:1px solid #E2E8F0;
        ">
    
            <div style="
                font-size:14px;
                color:#334155;
            ">
                • {item["dimension"]}
            </div>
    
            <div style="
                font-size:15px;
                font-weight:600;
                color:#DC2626;
            ">
                {item["score"]:.1f}
            </div>
    
        </div>
        """
    
    similar_col, profile_col = st.columns(
        [1, 1],
        gap="large"
    )
    
    
    with similar_col:
    
        st.html(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #D7E3F0;
                border-radius:16px;
                padding:24px;
                height:300px;
                box-sizing:border-box;
            ">
    
                <div style="
                    font-size:17px;
                    font-weight:600;
                    color:#1E293B;
                    margin-bottom:5px;
                ">
                    🧩 Países con perfil similar
                </div>
    
                <div style="
                    font-size:12px;
                    color:#64748B;
                    margin-bottom:14px;
                ">
                    Perfil del país seleccionado:
                    <b>{similar_results["profile"]}</b>
                </div>
    
                {similar_html}
    
            </div>
            """
        )
    
    
    with profile_col:
    
        st.html(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #D7E3F0;
                border-radius:16px;
                padding:24px;
                height:300px;
                box-sizing:border-box;
            ">
    
                <div style="
                    font-size:17px;
                    font-weight:600;
                    color:#1E293B;
                    margin-bottom:18px;
                ">
                    🔎 Perfil del país
                </div>
    
                <div style="
                    display:flex;
                    gap:35px;
                ">
    
                    <div style="
                        flex:1;
                    ">
    
                        <div style="
                            color:#16A34A;
                            font-size:14px;
                            font-weight:600;
                            margin-bottom:8px;
                        ">
                            🟢 Fortalezas relativas
                        </div>
    
                        {strengths_html}
    
                    </div>
    
    
                    <div style="
                        flex:1;
                    ">
    
                        <div style="
                            color:#DC2626;
                            font-size:14px;
                            font-weight:600;
                            margin-bottom:8px;
                        ">
                            🔴 Vulnerabilidades
                        </div>
    
                        {weaknesses_html}
    
                    </div>
    
                </div>
    
                <div style="
                    font-size:11px;
                    color:#94A3B8;
                    margin-top:18px;
                ">
                    Escala 0–100:
                    menor puntuación = menor vulnerabilidad.
                </div>
    
            </div>
            """
        )




    # Importancia de variables
    importance = get_model_importance(
        prediction_model,
        prediction_feature_names,
        top_n=10
    )

    st.markdown(
        """
        <style>
    
        .st-key-importance_prediction {
            background:#FFFFFF;
            border:1px solid #D7E3F0;
            border-radius:16px;
            padding:18px 22px 4px 22px;
            overflow:hidden;
            margin-top:18px;
        }
    
        .st-key-importance_prediction .stPlotlyChart {
            border-radius:16px;
            overflow:hidden;
        }
    
        </style>
        """,
        unsafe_allow_html=True
    )


    fig_importance = px.bar(
        importance,
        x="importance",
        y="variable",
        orientation="h"
    )
    
    fig_importance.update_layout(
        title=dict(
            text="Principales variables del modelo",
            x=0.01,
            xanchor="left"
        ),
        template="plotly_white",
        height=470,
        xaxis_title="Importancia",
        yaxis_title="",
        showlegend=False,
    
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    
        margin=dict(
            l=30,
            r=30,
            t=65,
            b=45
        )
    )
    
    
    with st.container(
        key="importance_prediction"
    ):
    
        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    pdf_bytes = create_prediction_report(
        country_name=selected_country_name,
        country_iso=selected_country_iso,
        reference_year=selected_prediction_year,
        target_year=target_year,
        prediction=prediction,
        equivalent_rating=equivalent_rating,
        risk_level=risk_level,
        current_rating=current_rating,
        change=change,
        agency_ratings=agency_ratings,
        interpretation=interpretation,
        similar_results=similar_results,
        country_risk_profile=country_risk_profile,
        fig_evolution=fig_evolution,
        fig_importance=fig_importance,
        n_features=len(prediction_features)
    )
    
    
    st.download_button(
        label="📄 Descargar informe PDF",
        data=pdf_bytes,
        file_name=(
            f"informe_prediccion_"
            f"{selected_country_iso}_"
            f"{target_year}.pdf"
        ),
        mime="application/pdf",
        use_container_width=False
    )

     # firma ------------------------
    st.markdown(
        """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
        m o-o l o-a l o
        </div>""",
        unsafe_allow_html=True
    )


    
#------------------------------------------------------------------------------------------
# Datos e indicadores
#------------------------------------------------------------------------------------------

elif page == "📊 Datos e indicadores":

    st.title(
        "Datos e indicadores"
    )

    st.markdown(
        """
        Consulta de los principales indicadores utilizados
        en el análisis del riesgo soberano.
        """
    )


    # Países disponibles
    countries_df = get_data_countries(
        cluster_data
    )

    country_options = dict(
        zip(
            countries_df["country"],
            countries_df["iso3"]
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        selected_data_country = st.selectbox(
            "🌍 País",
            list(
                country_options.keys()
            ),
            index=(
                list(
                    country_options.keys()
                ).index("Spain")
                if "Spain"
                in country_options
                else 0
            ),
            key="data_country"
        )


    selected_data_iso = (
        country_options[
            selected_data_country
        ]
    )


    available_data_years = get_data_years(
        cluster_data,
        selected_data_iso
    )


    with col2:

        selected_data_year = st.selectbox(
            "📅 Año",
            available_data_years,
            index=0,
            key="data_year"
        )


    # Formato de valores
    def format_indicator_value(
        variable,
        value
    ):

        if pd.isna(value):
            return "."

        percentage_variables = [
            "gdp_growth_pct",
            "inflation_cpi_pct",
            "unemployment_pct",
            "government_debt_pct_gdp",
            "fiscal_balance_pct_gdp",
            "current_account_balance_pct_gdp"
        ]

        if variable in percentage_variables:

            return f"{value:.1f}%"

        return f"{value:.1f}"


    # Indicadores principales
    key_indicators = get_key_indicators(
        cluster_data,
        selected_data_iso,
        selected_data_year
    )


    st.markdown(
        "### Indicadores principales"
    )


    card_cols = st.columns(6)


    icons = {
        "gdp_growth_pct": "📈",
        "inflation_cpi_pct": "🔥",
        "unemployment_pct": "👥",
        "government_debt_pct_gdp": "🏦",
        "fiscal_balance_pct_gdp": "⚖️",
        "current_account_balance_pct_gdp": "🌍"
    }


    for col, indicator in zip(
        card_cols,
        key_indicators
    ):

        value_text = format_indicator_value(
            indicator["variable"],
            indicator["value"]
        )

        icon = icons.get(
            indicator["variable"],
            "📊"
        )


        with col:

            st.html(
                f"""
                <div style="
                    background:#FFFFFF;
                    border:1px solid #D7E3F0;
                    border-radius:16px;
                    padding:20px 16px;
                    height:160px;
                    box-sizing:border-box;
                    display:flex;
                    flex-direction:column;
                ">

                    <div style="
                        font-size:14px;
                        font-weight:600;
                        color:#1E293B;
                        line-height:1.35;
                    ">
                        {icon}
                        {indicator["label"]}
                    </div>

                    <div style="
                        margin-top:auto;
                        text-align:center;
                        font-size:27px;
                        font-weight:700;
                        color:#1E293B;
                    ">
                        {value_text}
                    </div>

                </div>
                """
            )


    st.write("")


    # Detalle por bloques
    st.markdown(
        "### Indicadores por bloques"
    )


    tabs = st.tabs(
        [
            "Macroeconómico",
            "Fiscal",
            "Externo",
            "Institucional",
            "Política y estabilidad",
            "Historial crediticio"
        ]
    )


    blocks = [
        "Macroeconómico",
        "Fiscal",
        "Externo",
        "Institucional",
        "Política y estabilidad",
        "Historial crediticio"
    ]


    for tab, block in zip(
        tabs,
        blocks
    ):

        with tab:

            indicator_table = get_indicator_table(
                cluster_data,
                selected_data_iso,
                selected_data_year,
                block
            )


            indicator_table["Valor"] = (
                indicator_table["Valor"]
                .apply(
                    lambda x:
                    "."
                    if pd.isna(x)
                    else f"{x:,.2f}"
                )
            )


            st.dataframe(
                indicator_table,
                use_container_width=True,
                hide_index=True
            )


    st.write("")


    # Evolución histórica
    st.markdown(
        "### Evolución de los indicadores"
    )


    indicator_options = (
        get_indicator_options()
    )


    selected_indicator_label = (
        st.selectbox(
            "📊 Indicador",
            list(
                indicator_options.keys()
            ),
            key="evolution_indicator"
        )
    )


    selected_indicator = (
        indicator_options[
            selected_indicator_label
        ]
    )


    clean_indicator_label = (
        selected_indicator_label
        .split(
            " · ",
            1
        )[-1]
    )


    fig_indicator = (
        plot_indicator_evolution(
            data=cluster_data,
            country_iso=selected_data_iso,
            variable=selected_indicator,
            label=clean_indicator_label
        )
    )


    st.markdown(
        """
        <style>

        .st-key-indicator_evolution_card {
            background:#FFFFFF;
            border:1px solid #D7E3F0;
            border-radius:16px;
            padding:18px 22px 4px 22px;
            overflow:hidden;
        }

        .st-key-indicator_evolution_card
        .stPlotlyChart {
            border-radius:16px;
            overflow:hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    with st.container(
        key="indicator_evolution_card"
    ):

        st.plotly_chart(
            fig_indicator,
            use_container_width=True
        )


    st.write("")


    # Descargar datos
    country_download = (
        get_country_download_data(
            cluster_data,
            selected_data_iso
        )
    )


    csv_data = (
        country_download
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        label=(
            f"⬇️ Descargar datos de "
            f"{selected_data_country}"
        ),
        data=csv_data,
        file_name=(
            f"{selected_data_iso}_"
            f"country_risk_data.csv"
        ),
        mime="text/csv"
    )



     # firma ------------------------
    st.markdown(
        """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
        m o-o l o-a l o
        </div>""",
        unsafe_allow_html=True
    )


#-------------------------------------------------------------------------------------------
elif page == "🔔 Alertas":

    st.title(
        "Alertas de riesgo soberano"
    )

    st.markdown(
        """
        Identificación automática de señales de riesgo
        para el país y año seleccionados.
        """
    )


    # Países disponibles
    alert_countries = get_available_countries(
        prediction_panel
    )

    alert_country_options = dict(
        zip(
            alert_countries["country"],
            alert_countries["iso3"]
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        selected_alert_country = st.selectbox(
            "🌍 País",
            list(
                alert_country_options.keys()
            ),
            index=(
                list(
                    alert_country_options.keys()
                ).index("Spain")
                if "Spain"
                in alert_country_options
                else 0
            ),
            key="alert_country"
        )


    selected_alert_iso = (
        alert_country_options[
            selected_alert_country
        ]
    )


    alert_years = get_available_prediction_years(
        prediction_panel,
        selected_alert_iso
    )


    with col2:

        selected_alert_year = st.selectbox(
            "📅 Año",
            alert_years,
            index=0,
            key="alert_year"
        )


    # Predicción del país
    alert_prediction_result = predict_country(
        panel=prediction_panel,
        model=prediction_model,
        preprocessor=prediction_preprocessor,
        features=prediction_features,
        country_iso=selected_alert_iso,
        year=selected_alert_year
    )


    alert_prediction = (
        alert_prediction_result[
            "prediction"
        ]
    )

    alert_current_rating = (
        alert_prediction_result[
            "current_rating"
        ]
    )


    # Obtener alertas
    alert_results = get_country_alerts(
        panel=prediction_panel,
        raw_data=cluster_data,
        country_iso=selected_alert_iso,
        year=selected_alert_year,
        prediction=alert_prediction,
        current_rating=alert_current_rating
    )


    overall_level = (
        alert_results[
            "overall_level"
        ]
    )

    high_count = (
        alert_results[
            "high_count"
        ]
    )

    medium_count = (
        alert_results[
            "medium_count"
        ]
    )

    main_vulnerability = (
        alert_results[
            "main_vulnerability"
        ]
    )


    # Color general
    if overall_level == "Alta":

        level_color = "#DC2626"
        level_background = "#FEE2E2"

    elif overall_level == "Atención":

        level_color = "#D97706"
        level_background = "#FEF3C7"

    else:

        level_color = "#16A34A"
        level_background = "#DCFCE7"


    # Tarjetas resumen
    c1, c2, c3, c4 = st.columns(4)


    cards_alerts = [
        (
            c1,
            "🚨",
            "Nivel de alerta",
            overall_level,
            level_color,
            level_background
        ),
        (
            c2,
            "🔴",
            "Alertas altas",
            high_count,
            "#DC2626",
            "#FFFFFF"
        ),
        (
            c3,
            "🟠",
            "Alertas de atención",
            medium_count,
            "#D97706",
            "#FFFFFF"
        ),
        (
            c4,
            "⚠️",
            "Principal vulnerabilidad",
            main_vulnerability,
            "#1E293B",
            "#FFFFFF"
        )
    ]


    for (
        col,
        icon,
        title,
        value,
        value_color,
        background
    ) in cards_alerts:

        with col:

            st.html(
                f"""
                <div style="
                    background:{background};
                    border:1px solid #D7E3F0;
                    border-radius:16px;
                    padding:20px 22px;
                    height:170px;
                    box-sizing:border-box;
                    display:flex;
                    flex-direction:column;
                ">

                    <div style="
                        font-size:15px;
                        font-weight:600;
                        color:#1E293B;
                    ">
                        {icon} {title}
                    </div>

                    <div style="
                        margin-top:auto;
                        text-align:center;
                        font-size:25px;
                        font-weight:700;
                        color:{value_color};
                    ">
                        {value}
                    </div>

                </div>
                """
            )


    st.write("")


    # Alertas detectadas
    st.markdown(
        "### Alertas detectadas"
    )


    alerts = alert_results[
        "alerts"
    ]


    if len(alerts) == 0:

        st.html(
            """
            <div style="
                background:#F0FDF4;
                border:1px solid #BBF7D0;
                border-radius:16px;
                padding:24px;
                margin-top:10px;
            ">

                <div style="
                    font-size:17px;
                    font-weight:600;
                    color:#15803D;
                    margin-bottom:8px;
                ">
                    🟢 Sin alertas relevantes
                </div>

                <div style="
                    font-size:14px;
                    color:#475569;
                ">
                    No se detectan señales de riesgo relevantes
                    para el país y año seleccionados.
                </div>

            </div>
            """
        )


    else:

        for alert in alerts:

            if alert["severity"] == "Alta":

                alert_icon = "🔴"
                alert_color = "#DC2626"
                alert_background = "#FEF2F2"
                alert_border = "#FECACA"

            else:

                alert_icon = "🟠"
                alert_color = "#D97706"
                alert_background = "#FFFBEB"
                alert_border = "#FDE68A"


            st.html(
                f"""
                <div style="
                    background:{alert_background};
                    border:1px solid {alert_border};
                    border-radius:14px;
                    padding:18px 22px;
                    margin-bottom:12px;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        gap:20px;
                    ">

                        <div>

                            <div style="
                                font-size:16px;
                                font-weight:600;
                                color:{alert_color};
                                margin-bottom:5px;
                            ">
                                {alert_icon}
                                {alert["title"]}
                            </div>

                            <div style="
                                font-size:12px;
                                color:#64748B;
                                margin-bottom:8px;
                            ">
                                {alert["category"]}
                            </div>

                            <div style="
                                font-size:14px;
                                color:#475569;
                            ">
                                {alert["message"]}
                            </div>

                        </div>

                        <div style="
                            min-width:110px;
                            text-align:right;
                            font-size:18px;
                            font-weight:700;
                            color:{alert_color};
                        ">
                            {alert["value"]}
                        </div>

                    </div>

                </div>
                """
            )
    # firma ------------------------
        st.markdown(
            """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
            m o-o l o-a l o
            </div>""",
            unsafe_allow_html=True
        )
#-------------------------------------------------------------------------------------------

elif page == "📄 Informes":

    st.title("Informes")

    st.markdown(
        """
        Generación automática de un informe completo
        de riesgo soberano para el país seleccionado.
        """
    )

    # Países disponibles
    report_countries = get_available_countries(
        prediction_panel
    )

    report_country_options = dict(
        zip(
            report_countries["country"],
            report_countries["iso3"]
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_report_country = st.selectbox(
            "🌍 País",
            list(report_country_options.keys()),
            index=(
                list(
                    report_country_options.keys()
                ).index("Spain")
                if "Spain"
                in report_country_options
                else 0
            ),
            key="report_country"
        )

    selected_report_iso = report_country_options[
        selected_report_country
    ]

    report_years = get_available_prediction_years(
        prediction_panel,
        selected_report_iso
    )

    with col2:

        selected_report_year = st.selectbox(
            "📅 Año de referencia",
            report_years,
            index=0,
            key="report_year"
        )

    report_target_year = (
        selected_report_year + 1
    )

    with col3:

        st.selectbox(
            "📈 Año predicción",
            [report_target_year],
            key="report_target_year"
        )

    # Predicción
    report_prediction_result = predict_country(
        panel=prediction_panel,
        model=prediction_model,
        preprocessor=prediction_preprocessor,
        features=prediction_features,
        country_iso=selected_report_iso,
        year=selected_report_year
    )

    report_prediction = (
        report_prediction_result["prediction"]
    )

    report_current_rating = (
        report_prediction_result["current_rating"]
    )

    report_change = (
        report_prediction
        - report_current_rating
    )

    report_equivalent_rating = score_to_rating(
        report_prediction
    )

    report_risk_level = get_risk_level(
        report_prediction
    )

    # Ratings agencias
    report_agency_ratings = get_agency_ratings(
        prediction_agencies,
        selected_report_iso,
        selected_report_year
    )
    # Interpretación
    report_interpretation = (
        get_prediction_interpretation(
            report_prediction,
            report_current_rating
        )
    )

    # Evolución
    report_fig_evolution = (
        plot_prediction_evolution(
            panel=prediction_panel,
            country_iso=selected_report_iso,
            year=selected_report_year,
            prediction=report_prediction
        )
    )

    # Importancia del modelo
    report_fig_importance = (
        get_model_importance(
            model=prediction_model,
            feature_names=prediction_feature_names,
            top_n=10
        )
    )

    # Indicadores principales
    report_key_indicators = (
        get_key_indicators(
            cluster_data,
            selected_report_iso,
            selected_report_year
        )
    )

    # Clúster
    report_cluster_results = (
        run_cluster_analysis(
            data=cluster_data,
            ratings=cluster_ratings,
            year=selected_report_year,
            region=None,
            selected_k=None
        )
    )

    # Países similares
    report_similar_results = (
        get_similar_countries(
            cluster_results=report_cluster_results,
            country_iso=selected_report_iso,
            panel=prediction_panel,
            model=prediction_model,
            preprocessor=prediction_preprocessor,
            features=prediction_features,
            year=selected_report_year,
            top_n=3
        )
    )

    # Fortalezas y vulnerabilidades
    report_risk_profile = (
        get_country_risk_profile(
            data=cluster_data,
            country_iso=selected_report_iso,
            year=selected_report_year,
            top_n=2
        )
    )

    # Alertas
    report_alert_results = (
        get_country_alerts(
            panel=prediction_panel,
            raw_data=cluster_data,
            country_iso=selected_report_iso,
            year=selected_report_year,
            prediction=report_prediction,
            current_rating=report_current_rating
        )
    )

    # Resumen visual
    st.markdown("### Resumen del informe")

    c1, c2, c3, c4 = st.columns(4)

    summary_cards = [
        (
            c1,
            "📈",
            "Predicción",
            f"{report_prediction:.1f}/100"
        ),
        (
            c2,
            "🏷️",
            "Rating estimado",
            report_equivalent_rating
        ),
        (
            c3,
            "🚨",
            "Nivel de alerta",
            report_alert_results[
                "overall_level"
            ]
        ),
        (
            c4,
            "🧩",
            "Perfil de riesgo",
            report_similar_results.get(
                "profile",
                "."
            )
        )
    ]

    for col, icon, title, value in summary_cards:

        with col:

            st.html(
                f"""
                <div style="
                    background:#FFFFFF;
                    border:1px solid #D7E3F0;
                    border-radius:16px;
                    padding:20px;
                    height:150px;
                    box-sizing:border-box;
                    display:flex;
                    flex-direction:column;
                ">

                    <div style="
                        font-size:15px;
                        font-weight:600;
                        color:#1E293B;
                    ">
                        {icon} {title}
                    </div>

                    <div style="
                        margin-top:auto;
                        text-align:center;
                        font-size:25px;
                        font-weight:700;
                        color:#1E293B;
                    ">
                        {value}
                    </div>

                </div>
                """
            )

    st.write("")

    # Contenido del informe
    st.html(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #D7E3F0;
            border-radius:16px;
            padding:24px;
        ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:#1E293B;
                margin-bottom:12px;
            ">
                📄 Informe soberano · {selected_report_country}
            </div>

            <div style="
                font-size:14px;
                color:#475569;
                line-height:1.8;
            ">

                El informe incluye:

                <br>• Predicción del rating soberano para {report_target_year}
                <br>• Ratings S&P, Moody's y Fitch
                <br>• Evolución histórica del rating
                <br>• Indicadores macroeconómicos, fiscales y externos
                <br>• Alertas de riesgo
                <br>• Fortalezas y vulnerabilidades
                <br>• Posicionamiento por clúster
                <br>• Países con perfil similar
                <br>• Principales variables del modelo
                <br>• Ficha metodológica

            </div>

        </div>
        """
    )

    st.write("")

    # Generar PDF
    report_pdf = create_full_country_report(
        country_name=selected_report_country,
        country_iso=selected_report_iso,
        reference_year=selected_report_year,
        target_year=report_target_year,
        prediction=report_prediction,
        equivalent_rating=report_equivalent_rating,
        risk_level=report_risk_level,
        current_rating=report_current_rating,
        change=report_change,
        agency_ratings=report_agency_ratings,
        interpretation=report_interpretation,
        key_indicators=report_key_indicators,
        alert_results=report_alert_results,
        similar_results=report_similar_results,
        country_risk_profile=report_risk_profile,
        cluster_results=report_cluster_results,
        fig_evolution=report_fig_evolution,
        fig_importance=report_fig_importance,
        n_features=len(
            prediction_features
        )
    )

    download_col1, download_col2 = (
        st.columns(2)
    )

    with download_col1:

        st.download_button(
            label="📄 Descargar informe completo PDF",
            data=report_pdf,
            file_name=(
                f"informe_riesgo_soberano_"
                f"{selected_report_iso}_"
                f"{report_target_year}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )

    # Descargar datos históricos
    report_country_data = (
        get_country_download_data(
            cluster_data,
            selected_report_iso
        )
    )

    report_csv = (
        report_country_data
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    with download_col2:

        st.download_button(
            label="📊 Descargar datos del país CSV",
            data=report_csv,
            file_name=(
                f"datos_"
                f"{selected_report_iso}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

    # firma ------------------------
        st.markdown(
            """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
            m o-o l o-a l o
            </div>""",
            unsafe_allow_html=True
        )
#-------------------------------------------------------------------------------------------
elif page == "⚙️ Metodología":
    st.title("Metodología")
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {
            background-color: white !important;
            border: 1px solid #D7E3F0 !important;
            border-radius: 14px !important;
            overflow: hidden;
        }
    
        div[data-testid="stExpander"] details {
            background-color: white !important;
        }
    
        div[data-testid="stExpander"] summary {
            background-color: white !important;
            border-radius: 14px !important;
        }
    
        div[data-testid="stExpander"] details[open] summary {
            border-bottom: 1px solid #E2E8F0 !important;
        }
    
        div[data-testid="stExpander"] .streamlit-expanderContent {
            background-color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
            """
            Descripción de los principales procedimientos utilizados
            en la herramienta de análisis y predicción del riesgo soberano.
            """
        )
    
    # Resumen general
    st.markdown("### Resumen del modelo")

    c1, c2, c3, c4 = st.columns(4)

    methodology_cards = [
        (
            c1,
            "🌳",
            "Modelo final",
            "Random Forest"
        ),
        (
            c2,
            "📏",
            "Horizonte",
            "Rating t+1"
        ),
        (
            c3,
            "📉",
            "MAE test",
            "4.988"
        ),
        (
            c4,
            "📈",
            "R² test",
            "0.928"
        )
    ]

    for col, icon, title, value in methodology_cards:

        with col:

            st.html(
                f"""
                <div style="
                    background:#FFFFFF;
                    border:1px solid #D7E3F0;
                    border-radius:16px;
                    padding:20px;
                    height:150px;
                    box-sizing:border-box;
                    display:flex;
                    flex-direction:column;
                ">

                    <div style="
                        font-size:15px;
                        font-weight:600;
                        color:#1E293B;
                    ">
                        {icon} {title}
                    </div>

                    <div style="
                        margin-top:auto;
                        text-align:center;
                        font-size:24px;
                        font-weight:700;
                        color:#1E293B;
                    ">
                        {value}
                    </div>

                </div>
                """
            )

    st.write("")

    # Predicción
    with st.expander(
        "📏 Predicción del rating soberano",
        expanded=True
    ):

        st.markdown(
            """
            El modelo definitivo es un **Random Forest** entrenado
            para estimar el rating soberano del año siguiente.

            Por tanto:

            **Indicadores del año t → Rating soberano del año t+1**

            La variable objetivo es el rating medio obtenido a partir
            de las calificaciones disponibles de **S&P, Moody's y Fitch**.

            Las agencias que no disponen de rating para un determinado
            país y año no se consideran en el cálculo de la media.
            """
        )

    # Escala
    with st.expander(
        "🏷️ Escala del rating"
    ):

        st.markdown(
            """
            Las calificaciones crediticias se transforman a una
            escala numérica común de **0 a 100**.

            Una puntuación mayor representa una mejor calidad
            crediticia y, por tanto, un menor riesgo soberano.

            La aplicación presenta además una equivalencia aproximada
            con la escala de letras utilizada por S&P y Fitch.
            """
        )

    # Dimensiones
    with st.expander(
        "📊 Dimensiones de riesgo"
    ):

        st.markdown(
            """
            Para facilitar la interpretación de los indicadores se
            construyen seis dimensiones de riesgo:

            - Macroeconómica
            - Fiscal
            - Externa
            - Institucional
            - Política y estabilidad
            - Historial crediticio

            Los indicadores se normalizan anualmente en una escala
            de **0 a 100 de vulnerabilidad**.

            En estas dimensiones:

            **0 = menor vulnerabilidad**

            **100 = mayor vulnerabilidad**

            Los indicadores favorables se invierten para mantener
            una interpretación homogénea.
            """
        )

    # Clustering
    with st.expander(
        "🧩 Formación de clústeres"
    ):

        st.markdown(
            """
            Los países se agrupan utilizando sus variables
            fundamentales.

            Antes de aplicar el clustering:

            - se eliminan variables con exceso de valores ausentes;
            - se eliminan variables sin variabilidad;
            - se imputan los valores ausentes mediante la mediana;
            - las variables se estandarizan;
            - se aplica Análisis de Componentes Principales (PCA).

            Posteriormente se utiliza **K-Means** sobre los componentes
            principales retenidos.

            El número de clústeres se selecciona automáticamente
            entre distintas alternativas utilizando el
            **Silhouette Score**, evitando grupos excesivamente pequeños.

            Las etiquetas de riesgo del clúster se asignan después
            utilizando el nivel medio de rating de sus países.
            """
        )

    # Países similares
    with st.expander(
        "🌍 Países con perfil similar"
    ):

        st.markdown(
            """
            Los países comparables pertenecen al mismo clúster que
            el país seleccionado.

            Dentro de dicho clúster se identifican las economías más
            próximas en el espacio definido por los componentes
            principales.

            Esta comparación permite identificar países con estructuras
            macroeconómicas, fiscales, externas e institucionales
            relativamente similares.
            """
        )

    # Alertas
    with st.expander(
        "🚨 Sistema de alertas"
    ):

        st.markdown(
            """
            Las alertas no proceden de un modelo adicional.

            Se construyen mediante reglas de interpretación sobre:

            - deterioro previsto del rating;
            - dimensiones con vulnerabilidad elevada;
            - defaults recientes;
            - inflación elevada;
            - recesión;
            - déficits persistentes;
            - conflictos armados.

            Las alertas complementan la predicción y permiten destacar
            rápidamente posibles focos de riesgo.
            """
        )

    # Validación
    with st.expander(
        "🧪 Validación temporal"
    ):

        st.markdown(
            """
            Para evitar utilizar información futura durante el
            entrenamiento, la muestra se divide cronológicamente:

            - **Entrenamiento:** 2000–2018
            - **Validación:** 2019–2021
            - **Test:** 2022–2024

            El modelo se selecciona utilizando la muestra de validación
            y su comportamiento final se evalúa sobre el periodo de test.
            """
        )

    # Nota final
    st.write("")

    st.info(
        """
        Los resultados de la herramienta tienen carácter analítico
        y académico. Las predicciones y perfiles de riesgo no
        constituyen una calificación crediticia emitida por una
        agencia de rating ni una recomendación de inversión.
        """
    )

    st.markdown(
        """<div style="text-align:center;margin-top:50px;color:#B0B7C3;font-family:monospace;letter-spacing:3px;font-size:11px;">
        m o-o l o-a l o
        </div>""",
        unsafe_allow_html=True
    )



     