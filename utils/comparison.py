import pandas as pd
import plotly.graph_objects as go


comparison_indicators = {
    "PIB real (%)": "gdp_growth_pct",
    "PIB per cápita PPA":
        "gdp_per_capita_ppp_constant_2021_intl_usd",
    "Inflación (%)": "inflation_cpi_pct",
    "Desempleo (%)": "unemployment_pct",
    "Deuda pública (% PIB)": "government_debt_pct_gdp",
    "Saldo fiscal (% PIB)": "fiscal_balance_pct_gdp",
    "Cuenta corriente (% PIB)":
        "current_account_balance_pct_gdp",
    "Reservas (meses importaciones)":
        "reserves_months_imports",
    "Efectividad gubernamental":
        "government_effectiveness_estimate",
    "Calidad regulatoria":
        "regulatory_quality_estimate"
}


indicator_direction = {
    "PIB real (%)": "higher_better",
    "PIB per cápita PPA": "higher_better",
    "Inflación (%)": "lower_better",
    "Desempleo (%)": "lower_better",
    "Deuda pública (% PIB)": "lower_better",
    "Saldo fiscal (% PIB)": "higher_better",
    "Cuenta corriente (% PIB)": "higher_better",
    "Reservas (meses importaciones)": "higher_better",
    "Efectividad gubernamental": "higher_better",
    "Calidad regulatoria": "higher_better"
}

# RADAR
def plot_risk_radar(
    risk_data,
    country_a,
    country_b
):

    dimensions = [
        "Macroeconómica",
        "Fiscal",
        "Externa",
        "Institucional",
        "Política y estabilidad",
        "Historial crediticio"
    ]

    display_dimensions = [
        "Macro",
        "Fiscal",
        "Externo",
        "Institucional",
        "Política",
        "Historial cred."
    ]

    country_a_data = (
        risk_data[
            risk_data["iso3"] == country_a
        ]
        .iloc[0]
    )

    country_b_data = (
        risk_data[
            risk_data["iso3"] == country_b
        ]
        .iloc[0]
    )

    values_a = (
        country_a_data[dimensions]
        .astype(float)
        .tolist()
    )

    values_b = (
        country_b_data[dimensions]
        .astype(float)
        .tolist()
    )

    # Cerrar el hexágono
    values_a_closed = (
        values_a
        + [values_a[0]]
    )

    values_b_closed = (
        values_b
        + [values_b[0]]
    )

    display_dimensions_closed = (
        display_dimensions
        + [display_dimensions[0]]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_a_closed,
            theta=display_dimensions_closed,
            fill="toself",
            name=country_a_data["country"],
            line=dict(
                color="#258BD2",
                width=3
            ),
            marker=dict(
                color="#258BD2",
                size=7
            ),
            fillcolor="rgba(37, 139, 210, 0.18)"
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=values_b_closed,
            theta=display_dimensions_closed,
            fill="toself",
            name=country_b_data["country"],
            line=dict(
                color="#20B26B",
                width=3
            ),
            marker=dict(
                color="#20B26B",
                size=7
            ),
            fillcolor="rgba(32, 178, 107, 0.16)"
        )
    )

    fig.update_layout(

        polar=dict(
    
            # Reservar espacio arriba para título y leyenda
            domain=dict(
                y=[0.00, 0.82]
            ),
    
            gridshape="linear",
    
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[
                    0,
                    25,
                    50,
                    75,
                    100
                ],
                angle=15,
                tickfont=dict(
                    size=10,
                    color="#64748B"
                ),
                gridcolor="#D8E0EA",
                gridwidth=1,
                linecolor="#CBD5E1"
            ),
    
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                gridcolor="#D8E0EA",
                linecolor="#CBD5E1",
                tickfont=dict(
                    size=14,
                    color="#1E293B",
                    family="Arial Black"
                )
            ),
    
            bgcolor="#F8FBFD"
        ),
    
        showlegend=True,
    
        # Leyenda debajo del título
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="bottom"
        ),
    
    
        height=550,
    
        margin=dict(
            l=60,
            r=60,
            t=25,
            b=50
        ),
    
        paper_bgcolor="white"

    )
    # Eliminar el título interno de Plotly
    # Eliminar definitivamente el título interno de Plotly
    fig.layout.title.text = ""

    return fig

# EVOLUCIÓN TEMPORAL
def plot_risk_evolution(
    ratings,
    country_data,
    country_a,
    country_b
):

    selected = (
        ratings[
            ratings["iso3"].isin(
                [country_a, country_b]
            )
        ]
        .copy()
    )

    country_names = (
        country_data[
            ["iso3", "country"]
        ]
        .drop_duplicates()
        .set_index("iso3")["country"]
        .to_dict()
    )

    fig = go.Figure()

    colors = {
        country_a: "#258BD2",
        country_b: "#20B26B"
    }

    for country in [
        country_a,
        country_b
    ]:

        country_values = (
            selected[
                selected["iso3"] == country
            ]
            .sort_values("year")
        )

        fig.add_trace(
            go.Scatter(
                x=country_values["year"],
                y=country_values["risk_score"],
                mode="lines+markers",
                name=country_names[country],
                line=dict(
                    color=colors[country],
                    width=3
                ),
                marker=dict(
                    size=6
                )
            )
        )

    fig.update_layout(
        title="Evolución del Risk Score",
        height=450,

        xaxis_title="",
        yaxis_title="Risk Score",

        yaxis=dict(
            range=[0, 100],
            gridcolor="#E8EDF3"
        ),

        xaxis=dict(
            gridcolor="#F1F5F9"
        ),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=50
        )
    )

    return fig

# TABLA COMPARATIVA
def create_comparison_table(
    data,
    year,
    country_a,
    country_b
):

    selected = (
        data[
            (data["year"] == year)
            & data["iso3"].isin(
                [country_a, country_b]
            )
        ]
        .copy()
    )

    names = (
        selected[
            ["iso3", "country"]
        ]
        .drop_duplicates()
        .set_index("iso3")["country"]
        .to_dict()
    )

    rows = []

    for label, variable in comparison_indicators.items():

        value_a = (
            selected.loc[
                selected["iso3"] == country_a,
                variable
            ]
            .iloc[0]
        )

        value_b = (
            selected.loc[
                selected["iso3"] == country_b,
                variable
            ]
            .iloc[0]
        )

        difference = value_a - value_b

        direction = indicator_direction[label]

        if difference > 0:

            arrow = "↑"

            if direction == "higher_better":
                status = "positive"
            else:
                status = "negative"

        elif difference < 0:

            arrow = "↓"

            if direction == "lower_better":
                status = "positive"
            else:
                status = "negative"

        else:

            arrow = "→"
            status = "neutral"

        rows.append({
            "Indicador": label,
            names[country_a]: value_a,
            names[country_b]: value_b,
            "Diferencia": difference,
            "Flecha": arrow,
            "Estado": status
        })

    return pd.DataFrame(rows)

def style_comparison_table(
    table,
    country_a_name,
    country_b_name
):

    table = table.copy()

    def format_difference(row):

        value = row["Diferencia"]
        arrow = row["Flecha"]

        if row["Estado"] == "positive":
            color = "#15803D"

        elif row["Estado"] == "negative":
            color = "#E63946"

        else:
            color = "#64748B"

        return (
            f'<span style="color:{color};'
            f'font-weight:700;">'
            f'{value:+.2f} {arrow}'
            f'</span>'
        )

    table["Diferencia"] = (
        table.apply(
            format_difference,
            axis=1
        )
    )

    table = table.drop(
        columns=[
            "Flecha",
            "Estado"
        ]
    )

    styled = (
        table.style

        .hide(axis="index")

        .format({
            country_a_name: "{:,.2f}",
            country_b_name: "{:,.2f}"
        })

        .set_properties(**{
            "font-size": "14px",
            "padding": "12px 16px",
            "border-bottom": "1px solid #E8EDF3"
        })

        .set_properties(
            subset=["Indicador"],
            **{
                "font-weight": "600",
                "text-align": "left",
                "color": "#111827"
            }
        )

        .set_properties(
            subset=[
                country_a_name,
                country_b_name
            ],
            **{
                "text-align": "center",
                "color": "#334155",
                "font-weight": "500"
            }
        )

        .set_properties(
            subset=["Diferencia"],
            **{
                "text-align": "center"
            }
        )

        .set_table_styles([

            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("table-layout", "fixed")
                ]
            },

            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#F8FAFC"),
                    ("font-weight", "700"),
                    ("padding", "14px"),
                    ("text-align", "center")
                ]
            },

            {
                "selector": "thead th:nth-child(1)",
                "props": [
                    ("width", "46%"),
                    ("text-align", "left")
                ]
            },

            {
                "selector":
                    "thead th:nth-child(2), "
                    "tbody td:nth-child(2)",
                "props": [
                    ("width", "18%"),
                    ("text-align", "center")
                ]
            },

            {
                "selector":
                    "thead th:nth-child(3), "
                    "tbody td:nth-child(3)",
                "props": [
                    ("width", "18%"),
                    ("text-align", "center")
                ]
            },

            {
                "selector":
                    "thead th:nth-child(4), "
                    "tbody td:nth-child(4)",
                "props": [
                    ("width", "18%"),
                    ("text-align", "center")
                ]
            },

            {
                "selector": "thead th:nth-child(2)",
                "props": [
                    ("color", "#258BD2"),
                    ("font-weight", "700"),
                    ("border-bottom",
                     "3px solid #258BD2")
                ]
            },

            {
                "selector": "thead th:nth-child(3)",
                "props": [
                    ("color", "#20B26B"),
                    ("font-weight", "700"),
                    ("border-bottom",
                     "3px solid #20B26B")
                ]
            },

            {
                "selector": "thead th:nth-child(4)",
                "props": [
                    ("color", "#111827"),
                    ("font-weight", "700"),
                    ("border-bottom",
                     "3px solid #94A3B8")
                ]
            },

            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#F8FBFD")
                ]
            }
        ])
    )

    return styled


# TARJETONES INICIO
def get_comparison_cards(
    data,
    ratings,
    year,
    country_a,
    country_b
):

    indicators_year = (
        data[
            (data["year"] == year)
            & data["iso3"].isin(
                [country_a, country_b]
            )
        ]
        .copy()
    )

    ratings_year = (
        ratings[
            (ratings["year"] == year)
            & ratings["iso3"].isin(
                [country_a, country_b]
            )
        ]
        .copy()
    )

    ratings_year["risk_score"] = (
        100
        - ratings_year["rating_score_mean"]
    )

    names = (
        indicators_year[
            ["iso3", "country"]
        ]
        .drop_duplicates()
        .set_index("iso3")["country"]
        .to_dict()
    )

    result = {}

    for country in [
        country_a,
        country_b
    ]:

        indicators_country = (
            indicators_year[
                indicators_year["iso3"]
                == country
            ]
            .iloc[0]
        )

        rating_country = (
            ratings_year[
                ratings_year["iso3"]
                == country
            ]
            .iloc[0]
        )

        result[country] = {

            "country":
                names[country],

            "Risk Score":
                rating_country[
                    "risk_score"
                ],

            "Rating":
                rating_country[
                    "rating_score_mean"
                ],

            "PIB real (%)":
                indicators_country[
                    "gdp_growth_pct"
                ],

            "Inflación (%)":
                indicators_country[
                    "inflation_cpi_pct"
                ],

            "Deuda pública (% PIB)":
                indicators_country[
                    "government_debt_pct_gdp"
                ]
        }

    return result


# LECTURA RÁPIDA AUTOMÁTICA
def create_quick_reading(
    risk_data,
    cards,
    country_a,
    country_b
):

    dimensions = [
        "Macroeconómica",
        "Fiscal",
        "Externa",
        "Institucional",
        "Política y estabilidad",
        "Historial crediticio"
    ]

    name_a = (
        cards[country_a]["country"]
    )

    name_b = (
        cards[country_b]["country"]
    )

    risk_a = (
        cards[country_a]["Risk Score"]
    )

    risk_b = (
        cards[country_b]["Risk Score"]
    )

    row_a = (
        risk_data[
            risk_data["iso3"] == country_a
        ]
        .iloc[0]
    )

    row_b = (
        risk_data[
            risk_data["iso3"] == country_b
        ]
        .iloc[0]
    )

    if risk_a < risk_b:

        global_text = (
            f"{name_a} presenta un menor riesgo soberano "
            f"que {name_b}, con un Risk Score de "
            f"{risk_a:.1f} frente a {risk_b:.1f}."
        )

    elif risk_b < risk_a:

        global_text = (
            f"{name_b} presenta un menor riesgo soberano "
            f"que {name_a}, con un Risk Score de "
            f"{risk_b:.1f} frente a {risk_a:.1f}."
        )

    else:

        global_text = (
            f"{name_a} y {name_b} presentan el mismo "
            f"Risk Score ({risk_a:.1f})."
        )

    differences = {}

    for dimension in dimensions:

        differences[dimension] = abs(
            row_a[dimension]
            - row_b[dimension]
        )

    largest_dimension = max(
        differences,
        key=differences.get
    )

    value_a = row_a[largest_dimension]
    value_b = row_b[largest_dimension]

    if value_a < value_b:

        dimension_text = (
            f"La principal diferencia se encuentra en la "
            f"dimensión {largest_dimension.lower()}, "
            f"donde {name_a} presenta menor vulnerabilidad "
            f"relativa que {name_b}."
        )

    elif value_b < value_a:

        dimension_text = (
            f"La principal diferencia se encuentra en la "
            f"dimensión {largest_dimension.lower()}, "
            f"donde {name_b} presenta menor vulnerabilidad "
            f"relativa que {name_a}."
        )

    else:

        dimension_text = ""

    return (
        global_text
        + " "
        + dimension_text
    )

# MAPA
def plot_comparison_map(
    country_data,
    country_a,
    country_b
):

    names = (
        country_data[
            ["iso3", "country"]
        ]
        .drop_duplicates()
        .set_index("iso3")["country"]
        .to_dict()
    )

    name_a = names[country_a]
    name_b = names[country_b]

    fig = go.Figure()

    fig.add_trace(
        go.Choropleth(
            locations=[country_a],
            z=[1],
            locationmode="ISO-3",
            colorscale=[
                [0, "#258BD2"],
                [1, "#258BD2"]
            ],
            showscale=False,
            name=name_a,
            showlegend=True,
            text=[name_a],
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            ),
            marker_line_color="white",
            marker_line_width=1
        )
    )

    fig.add_trace(
        go.Choropleth(
            locations=[country_b],
            z=[1],
            locationmode="ISO-3",
            colorscale=[
                [0, "#20B26B"],
                [1, "#20B26B"]
            ],
            showscale=False,
            name=name_b,
            showlegend=True,
            text=[name_b],
            hovertemplate=(
                "<b>%{text}</b>"
                "<extra></extra>"
            ),
            marker_line_color="white",
            marker_line_width=1
        )
    )

    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#CBD5E1",
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#D9DEE5",
        showocean=True,
        oceancolor="white",
        bgcolor="white"
    )

    fig.update_layout(
        title=dict(
            text="Localización de los países",
            x=0.03
        ),
        height=480,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.03
        ),
        margin=dict(
            l=5,
            r=5,
            t=70,
            b=5
        ),
        paper_bgcolor="white"
    )

    return fig



    