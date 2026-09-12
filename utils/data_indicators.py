import pandas as pd
import plotly.graph_objects as go


INDICATOR_GROUPS = {

    "Macroeconómico": {
        "gdp_growth_pct": "Crecimiento PIB real (%)",
        "gdp_per_capita_constant_2015_usd": "PIB per cápita (USD 2015)",
        "gdp_per_capita_ppp_constant_2021_intl_usd": "PIB per cápita PPA",
        "gross_savings_pct_gdp": "Ahorro bruto (% PIB)",
        "investment_pct_gdp": "Inversión (% PIB)",
        "inflation_cpi_pct": "Inflación (%)",
        "unemployment_pct": "Desempleo (%)",
        "real_interest_rate_pct": "Tipo de interés real (%)"
    },

    "Fiscal": {
        "government_debt_pct_gdp": "Deuda pública (% PIB)",
        "fiscal_balance_pct_gdp": "Saldo fiscal (% PIB)",
        "primary_balance_pct_gdp": "Saldo primario (% PIB)",
        "government_revenue_pct_gdp": "Ingresos públicos (% PIB)",
        "government_expenditure_pct_gdp": "Gasto público (% PIB)",
        "net_interest_payments_pct_gdp": "Pagos netos de intereses (% PIB)"
    },

    "Externo": {
        "current_account_balance_pct_gdp": "Cuenta corriente (% PIB)",
        "exports_growth_pct": "Crecimiento exportaciones (%)",
        "exports_pct_gdp": "Exportaciones (% PIB)",
        "imports_pct_gdp": "Importaciones (% PIB)",
        "external_debt_pct_gni": "Deuda externa (% RNB)",
        "external_debt_service_pct_exports": "Servicio deuda externa (% exportaciones)",
        "fdi_net_inflows_pct_gdp": "IED neta (% PIB)",
        "reserves_months_imports": "Reservas (meses de importaciones)",
        "reserves_pct_external_debt": "Reservas (% deuda externa)",
        "short_term_debt_pct_reserves": "Deuda corto plazo (% reservas)"
    },

    "Institucional": {
        "control_of_corruption_estimate": "Control de la corrupción",
        "government_effectiveness_estimate": "Efectividad gubernamental",
        "political_stability_estimate": "Estabilidad política",
        "regulatory_quality_estimate": "Calidad regulatoria",
        "rule_of_law_estimate": "Estado de derecho",
        "voice_accountability_estimate": "Voz y rendición de cuentas"
    },

    "Política y estabilidad": {
        "electoral_democracy_index": "Índice de democracia electoral",
        "head_of_government_tenure_years": "Años del jefe de gobierno",
        "average_government_duration_since_2000": "Duración media del gobierno",
        "government_changes_previous_5y": "Cambios de gobierno últimos 5 años",
        "fragile_states_index": "Fragile States Index",
        "battle_related_deaths": "Muertes relacionadas con conflictos",
        "armed_conflict_dummy": "Conflicto armado"
    },

    "Historial crediticio": {
        "sovereign_debt_in_default_usd_mn": "Deuda soberana en default",
        "default_previous_5y": "Default últimos 5 años",
        "default_previous_10y": "Default últimos 10 años",
        "years_since_last_default": "Años desde último default",
        "never_default_history_dummy": "Nunca ha entrado en default"
    }
}


KEY_INDICATORS = {
    "gdp_growth_pct": "PIB real",
    "inflation_cpi_pct": "Inflación",
    "unemployment_pct": "Desempleo",
    "government_debt_pct_gdp": "Deuda pública",
    "fiscal_balance_pct_gdp": "Saldo fiscal",
    "current_account_balance_pct_gdp": "Cuenta corriente"
}


def get_data_countries(data):

    countries = (
        data[
            ["iso3", "country"]
        ]
        .drop_duplicates()
        .dropna()
        .sort_values("country")
        .reset_index(drop=True)
    )

    return countries


def get_data_years(
    data,
    country_iso
):

    years = (
        data[
            data["iso3"] == country_iso
        ]["year"]
        .dropna()
        .astype(int)
        .sort_values(
            ascending=False
        )
        .unique()
        .tolist()
    )

    return years


def get_country_year_row(
    data,
    country_iso,
    year
):

    row = data[
        (data["iso3"] == country_iso)
        & (data["year"] == year)
    ].copy()

    if row.empty:
        return None

    return row.iloc[0]


def get_key_indicators(
    data,
    country_iso,
    year
):

    row = get_country_year_row(
        data,
        country_iso,
        year
    )

    if row is None:
        return []

    results = []

    for variable, label in KEY_INDICATORS.items():

        if variable not in row.index:
            continue

        value = row[variable]

        results.append({
            "variable": variable,
            "label": label,
            "value": value
        })

    return results


def get_indicator_table(
    data,
    country_iso,
    year,
    block
):

    row = get_country_year_row(
        data,
        country_iso,
        year
    )

    if row is None:
        return pd.DataFrame()

    indicators = INDICATOR_GROUPS[
        block
    ]

    rows = []

    for variable, label in indicators.items():

        if variable not in row.index:
            continue

        value = row[variable]

        rows.append({
            "Indicador": label,
            "Valor": value
        })

    table = pd.DataFrame(
        rows
    )

    return table


def get_indicator_options():

    options = {}

    for block, indicators in INDICATOR_GROUPS.items():

        for variable, label in indicators.items():

            options[
                f"{block} · {label}"
            ] = variable

    return options


def plot_indicator_evolution(
    data,
    country_iso,
    variable,
    label
):

    country_data = (
        data[
            data["iso3"] == country_iso
        ]
        .sort_values("year")
        .copy()
    )

    if country_data.empty:
        return go.Figure()

    region = (
        country_data[
            "region"
        ]
        .dropna()
    )

    if len(region) > 0:
        region = region.iloc[0]
    else:
        region = None


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=country_data["year"],
            y=country_data[variable],
            mode="lines",
            name=country_data["country"].iloc[0],
            line=dict(
                color="#8EC5FF",
                width=3
            )
        )
    )


    if region is not None:

        region_data = (
            data[
                data["region"] == region
            ]
            .groupby(
                "year",
                as_index=False
            )[variable]
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=region_data["year"],
                y=region_data[variable],
                mode="lines",
                name="Media regional",
                line=dict(
                    color="#A0A0A0",
                    width=2,
                    dash="dash"
                )
            )
        )


    fig.update_layout(
        title=f"Evolución de {label}",
        height=430,
        xaxis_title="Año",
        yaxis_title=label,
        legend=dict(
            orientation="h",
            y=-0.18
        ),
        margin=dict(
            l=45,
            r=25,
            t=60,
            b=65
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )


    return fig


def get_country_download_data(
    data,
    country_iso
):

    country_data = (
        data[
            data["iso3"] == country_iso
        ]
        .sort_values("year")
        .copy()
    )

    return country_data