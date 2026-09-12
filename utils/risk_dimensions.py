import numpy as np
import pandas as pd

import numpy as np


dimension_variables = {

    "Macroeconómica": [
        "gdp_growth_pct",
        "gdp_per_capita_ppp_constant_2021_intl_usd",
        "inflation_cpi_pct",
        "unemployment_pct",
        "investment_pct_gdp",
        "gross_savings_pct_gdp"
    ],

    "Fiscal": [
        "government_debt_pct_gdp",
        "fiscal_balance_pct_gdp",
        "primary_balance_pct_gdp",
        "net_interest_payments_pct_gdp"
    ],

    "Externa": [
        "current_account_balance_pct_gdp",
        "reserves_months_imports",
        "fdi_net_inflows_pct_gdp"
    ],

    "Institucional": [
        "government_effectiveness_estimate",
        "regulatory_quality_estimate",
        "rule_of_law_estimate",
        "control_of_corruption_estimate",
        "voice_accountability_estimate"
    ],

    "Política y estabilidad": [
        "political_stability_estimate",
        "electoral_democracy_index",
        "armed_conflict_dummy"
    ],

    "Historial crediticio": [
        "years_since_last_default",
        "never_default_history_dummy",
        "default_previous_5y",
        "current_default_dummy"
    ]
}


higher_value_higher_risk = [
    "inflation_cpi_pct",
    "unemployment_pct",
    "government_debt_pct_gdp",
    "net_interest_payments_pct_gdp",
    "default_previous_5y",
    "current_default_dummy",
    "armed_conflict_dummy"
]


higher_value_lower_risk = [
    "gdp_growth_pct",
    "gdp_per_capita_ppp_constant_2021_intl_usd",
    "investment_pct_gdp",
    "gross_savings_pct_gdp",
    "fiscal_balance_pct_gdp",
    "primary_balance_pct_gdp",
    "current_account_balance_pct_gdp",
    "reserves_months_imports",
    "fdi_net_inflows_pct_gdp",
    "government_effectiveness_estimate",
    "regulatory_quality_estimate",
    "rule_of_law_estimate",
    "control_of_corruption_estimate",
    "voice_accountability_estimate",
    "political_stability_estimate",
    "electoral_democracy_index",
    "years_since_last_default",
    "never_default_history_dummy"
]

def calculate_percentile_risk(
    data,
    variable,
    higher_is_risk=True
):

    percentile = (
        data[variable]
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )

    if higher_is_risk:
        return percentile

    return 100 - percentile


def calculate_risk_dimensions(
    data,
    year
):

    year_data = (
        data[
            data["year"] == year
        ]
        .copy()
    )

    # Crear indicador de default actual
    year_data["current_default_dummy"] = (
        year_data[
            "sovereign_debt_in_default_usd_mn"
        ]
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    # Variables donde un valor mayor implica más riesgo
    for variable in higher_value_higher_risk:

        if variable in [
            "default_previous_5y",
            "current_default_dummy",
            "armed_conflict_dummy"
        ]:
            continue

        year_data[
            "risk_" + variable
        ] = calculate_percentile_risk(
            year_data,
            variable,
            higher_is_risk=True
        )

    # Variables donde un valor mayor implica menos riesgo
    for variable in higher_value_lower_risk:

        if variable == "never_default_history_dummy":
            continue

        year_data[
            "risk_" + variable
        ] = calculate_percentile_risk(
            year_data,
            variable,
            higher_is_risk=False
        )

    # Variables binarias
    year_data[
        "risk_default_previous_5y"
    ] = (
        year_data[
            "default_previous_5y"
        ]
        .map({
            0: 0,
            1: 100
        })
    )

    year_data[
        "risk_current_default_dummy"
    ] = (
        year_data[
            "current_default_dummy"
        ]
        .map({
            0: 0,
            1: 100
        })
    )

    year_data[
        "risk_never_default_history_dummy"
    ] = (
        year_data[
            "never_default_history_dummy"
        ]
        .map({
            0: 100,
            1: 0
        })
    )

    year_data[
        "risk_armed_conflict_dummy"
    ] = (
        year_data[
            "armed_conflict_dummy"
        ]
        .map({
            0: 0,
            1: 100
        })
    )

    # Calcular dimensiones
    for dimension, variables in (
        dimension_variables.items()
    ):

        risk_variables = [
            "risk_" + variable
            for variable in variables
        ]

        available_count = (
            year_data[
                risk_variables
            ]
            .notna()
            .sum(axis=1)
        )

        minimum_required = int(
            np.ceil(
                len(risk_variables)
                * 0.5
            )
        )

        dimension_score = (
            year_data[
                risk_variables
            ]
            .mean(axis=1)
        )

        dimension_score[
            available_count
            < minimum_required
        ] = np.nan

        year_data[
            dimension
        ] = dimension_score

    return year_data








