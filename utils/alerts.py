import pandas as pd

from utils.risk_dimensions import (
    calculate_risk_dimensions
)


DIMENSION_COLUMNS = [
    "Macroeconómica",
    "Fiscal",
    "Externa",
    "Institucional",
    "Política y estabilidad",
    "Historial crediticio"
]


def get_country_alerts(
    panel,
    raw_data,
    country_iso,
    year,
    prediction=None,
    current_rating=None
):

    alerts = []

    country_row = panel[
        (panel["iso3"] == country_iso)
        & (panel["year"] == year)
    ].copy()

    if country_row.empty:

        return {
            "alerts": [],
            "overall_level": "Sin datos",
            "high_count": 0,
            "medium_count": 0,
            "main_vulnerability": "."
        }

    row = country_row.iloc[0]


    # Predicción frente al rating actual
    if (
        prediction is not None
        and current_rating is not None
        and pd.notna(current_rating)
    ):

        change = (
            prediction
            - current_rating
        )

        if change <= -3:

            alerts.append({
                "severity": "Alta",
                "category": "Crediticio",
                "title": "Deterioro crediticio previsto",
                "message": (
                    "El modelo anticipa una caída relevante "
                    "respecto al rating medio actual."
                ),
                "value": f"{change:.1f} puntos"
            })

        elif change <= -1:

            alerts.append({
                "severity": "Atención",
                "category": "Crediticio",
                "title": "Ligero deterioro previsto",
                "message": (
                    "La predicción se sitúa por debajo "
                    "del rating medio actual."
                ),
                "value": f"{change:.1f} puntos"
            })


    # Dimensiones de riesgo
    risk_dimensions = calculate_risk_dimensions(
        raw_data,
        year
    )

    country_dimensions = risk_dimensions[
        risk_dimensions["iso3"] == country_iso
    ].copy()


    main_vulnerability = "."


    if not country_dimensions.empty:

        dimension_values = (
            country_dimensions[
                DIMENSION_COLUMNS
            ]
            .iloc[0]
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        if len(dimension_values) > 0:

            main_vulnerability = (
                dimension_values.index[0]
            )


        for dimension, value in (
            dimension_values.items()
        ):

            if value >= 75:

                alerts.append({
                    "severity": "Alta",
                    "category": "Dimensión de riesgo",
                    "title": (
                        f"Vulnerabilidad elevada: "
                        f"{dimension}"
                    ),
                    "message": (
                        "La dimensión se sitúa en la zona "
                        "de mayor vulnerabilidad relativa."
                    ),
                    "value": f"{value:.1f}/100"
                })

            elif value >= 60:

                alerts.append({
                    "severity": "Atención",
                    "category": "Dimensión de riesgo",
                    "title": (
                        f"Atención en: "
                        f"{dimension}"
                    ),
                    "message": (
                        "La dimensión presenta una "
                        "vulnerabilidad superior a la media."
                    ),
                    "value": f"{value:.1f}/100"
                })


    # Default reciente
    if (
        "default_previous_5y" in row.index
        and row["default_previous_5y"] == 1
    ):

        alerts.append({
            "severity": "Alta",
            "category": "Crediticio",
            "title": "Default soberano reciente",
            "message": (
                "El país registra un episodio de default "
                "en los últimos cinco años."
            ),
            "value": "Sí"
        })


    # Inflación elevada
    if (
        "high_inflation_dummy" in row.index
        and row["high_inflation_dummy"] == 1
    ):

        alerts.append({
            "severity": "Atención",
            "category": "Macroeconómico",
            "title": "Inflación elevada",
            "message": (
                "El indicador de inflación elevada "
                "se encuentra activo."
            ),
            "value": (
                f"{row['inflation_cpi_pct']:.1f}%"
                if pd.notna(
                    row.get(
                        "inflation_cpi_pct"
                    )
                )
                else "Sí"
            )
        })


    # Recesión
    if (
        "recession_dummy" in row.index
        and row["recession_dummy"] == 1
    ):

        alerts.append({
            "severity": "Atención",
            "category": "Macroeconómico",
            "title": "Contracción económica",
            "message": (
                "El país presenta crecimiento "
                "económico negativo."
            ),
            "value": (
                f"{row['gdp_growth_pct']:.1f}%"
                if pd.notna(
                    row.get(
                        "gdp_growth_pct"
                    )
                )
                else "Sí"
            )
        })


    # Déficit fiscal persistente
    if (
        "persistent_deficit_dummy" in row.index
        and row["persistent_deficit_dummy"] == 1
    ):

        alerts.append({
            "severity": "Atención",
            "category": "Fiscal",
            "title": "Déficit fiscal persistente",
            "message": (
                "El saldo fiscal muestra un patrón "
                "de déficit persistente."
            ),
            "value": (
                f"{row['fiscal_balance_pct_gdp']:.1f}% PIB"
                if pd.notna(
                    row.get(
                        "fiscal_balance_pct_gdp"
                    )
                )
                else "Sí"
            )
        })


    # Déficit exterior persistente
    if (
        "persistent_current_account_deficit"
        in row.index
        and row[
            "persistent_current_account_deficit"
        ] == 1
    ):

        alerts.append({
            "severity": "Atención",
            "category": "Externo",
            "title": "Déficit exterior persistente",
            "message": (
                "La cuenta corriente presenta "
                "un déficit persistente."
            ),
            "value": (
                f"{row['current_account_balance_pct_gdp']:.1f}% PIB"
                if pd.notna(
                    row.get(
                        "current_account_balance_pct_gdp"
                    )
                )
                else "Sí"
            )
        })


    # Conflicto armado
    if (
        "armed_conflict_dummy" in row.index
        and row["armed_conflict_dummy"] == 1
    ):

        alerts.append({
            "severity": "Alta",
            "category": "Político",
            "title": "Conflicto armado",
            "message": (
                "Existe registro de conflicto armado "
                "para el año seleccionado."
            ),
            "value": "Sí"
        })


    # Ordenar alertas
    severity_order = {
        "Alta": 0,
        "Atención": 1
    }

    alerts = sorted(
        alerts,
        key=lambda x:
        severity_order.get(
            x["severity"],
            2
        )
    )


    high_count = sum(
        alert["severity"] == "Alta"
        for alert in alerts
    )

    medium_count = sum(
        alert["severity"] == "Atención"
        for alert in alerts
    )


    if high_count > 0:

        overall_level = "Alta"

    elif medium_count > 0:

        overall_level = "Atención"

    else:

        overall_level = "Sin alertas relevantes"


    return {
        "alerts": alerts,
        "overall_level": overall_level,
        "high_count": high_count,
        "medium_count": medium_count,
        "main_vulnerability": main_vulnerability
    }