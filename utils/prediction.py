from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.risk_dimensions import calculate_risk_dimensions


from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = next(
    parent
    for parent in CURRENT_FILE.parents
    if (parent / "data").exists()
)

MODEL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "model"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


FINAL_MODEL_FILE = (
    MODEL_DATA_PATH
    / "random_forest_final.joblib"
)

PREPARED_DATA_FILE = (
    MODEL_DATA_PATH
    / "notebook_4_1_prepared_data.joblib"
)

AGENCY_RATINGS_FILE = (
    PROCESSED_PATH
    / "government_credit_ratings_annual_by_agency_2000_2025.csv"
)


def load_prediction_data():

    model_package = joblib.load(
        FINAL_MODEL_FILE
    )

    prepared_data = joblib.load(
        PREPARED_DATA_FILE
    )

    panel = pd.concat(
        [
            prepared_data["train"],
            prepared_data["validation"],
            prepared_data["test"]
        ],
        axis=0
    ).copy()

    panel = panel.sort_values(
        ["iso3", "year"]
    ).reset_index(drop=True)

    agency_ratings = pd.read_csv(
        AGENCY_RATINGS_FILE,
        low_memory=False
    )

    return {
        "model": model_package["model"],
        "preprocessor": model_package["preprocessor"],
        "features": model_package["features"],
        "feature_names": model_package["feature_names"],
        "panel": panel,
        "agency_ratings": agency_ratings
    }


def predict_country(
    panel,
    model,
    preprocessor,
    features,
    country_iso,
    year
):

    row = panel[
        (panel["iso3"] == country_iso)
        & (panel["year"] == year)
    ].copy()

    if row.empty:
        raise ValueError(
            "No hay datos disponibles para "
            "el país y año seleccionados."
        )

    X = row[features]

    X_processed = preprocessor.transform(
        X
    )

    prediction = float(
        model.predict(
            X_processed
        )[0]
    )

    prediction = np.clip(
        prediction,
        0,
        100
    )

    current_rating = row[
        "rating_score_mean_t"
    ].iloc[0]

    if pd.isna(current_rating):

        change = np.nan

    else:

        change = (
            prediction
            - float(current_rating)
        )

    return {
        "country": row["country"].iloc[0],
        "iso3": country_iso,
        "year": int(year),
        "target_year": int(year) + 1,
        "prediction": prediction,
        "current_rating": current_rating,
        "change": change,
        "row": row
    }


def score_to_rating(
    score
):

    if pd.isna(score):
        return "."

    scale = [
        (97.5, "AAA"),
        (92.5, "AA+"),
        (87.5, "AA"),
        (82.5, "AA-"),
        (77.5, "A+"),
        (72.5, "A"),
        (67.5, "A-"),
        (62.5, "BBB+"),
        (57.5, "BBB"),
        (52.5, "BBB-"),
        (47.5, "BB+"),
        (42.5, "BB"),
        (37.5, "BB-"),
        (32.5, "B+"),
        (27.5, "B"),
        (22.5, "B-"),
        (17.5, "CCC+"),
        (12.5, "CCC"),
        (7.5, "CCC-"),
        (2.5, "CC")
    ]

    for limit, rating in scale:

        if score >= limit:
            return rating

    return "D"


def get_risk_level(
    score
):

    if score >= 80:
        return "Muy bajo"

    elif score >= 65:
        return "Bajo"

    elif score >= 45:
        return "Medio"

    elif score >= 25:
        return "Alto"

    else:
        return "Muy alto"


def get_agency_ratings(
    agency_ratings,
    country_iso,
    year
):

    ratings = agency_ratings[
        (agency_ratings["iso3"] == country_iso)
        & (agency_ratings["year"] == year)
        & (
            agency_ratings[
                "rating_status"
            ] == "active"
        )
    ].copy()

    result = {
        "S&P": ".",
        "Moody's": ".",
        "Fitch": "."
    }

    for agency in result:

        value = ratings.loc[
            ratings["agency"] == agency,
            "rating_grade"
        ]

        if not value.empty:

            rating = value.iloc[0]

            if pd.notna(rating):
                result[agency] = str(
                    rating
                )

    return result


def get_prediction_interpretation(
    prediction,
    current_rating
):

    if pd.isna(current_rating):

        return {
            "status": "Sin comparación",
            "icon": "⚪",
            "text": (
                "No existe un rating actual disponible "
                "para calcular la variación."
            )
        }

    change = (
        prediction
        - current_rating
    )

    if change > 3:

        return {
            "status": "Mejora esperada",
            "icon": "🟢",
            "text": (
                "El modelo estima una mejora del perfil "
                "crediticio respecto a la situación actual."
            )
        }

    elif change < -3:

        return {
            "status": "Deterioro esperado",
            "icon": "🔴",
            "text": (
                "El modelo anticipa un deterioro del perfil "
                "crediticio respecto a la situación actual."
            )
        }

    else:

        return {
            "status": "Estabilidad esperada",
            "icon": "🟡",
            "text": (
                "El modelo estima una situación relativamente "
                "estable respecto al rating actual."
            )
        }


def plot_prediction_evolution(
    panel,
    country_iso,
    year,
    prediction
):

    country_history = panel[
        (panel["iso3"] == country_iso)
        & (panel["year"] <= year)
    ].copy()

    country_history = (
        country_history
        .sort_values(
            "year"
        )
    )

    if country_history.empty:

        raise ValueError(
            "No existe histórico disponible "
            "para el país seleccionado."
        )

    region = (
        country_history[
            "region"
        ]
        .iloc[0]
    )

    region_history = (
        panel[
            (panel["region"] == region)
            & (panel["year"] <= year)
        ]
        .groupby(
            "year",
            as_index=False
        )["rating_score_mean_t"]
        .mean()
    )

    country_name = (
        country_history[
            "country"
        ]
        .iloc[0]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=country_history[
                "year"
            ],
            y=country_history[
                "rating_score_mean_t"
            ],
            mode="lines",
            name=country_name,
            line=dict(
                color="#8EC5FF",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=region_history[
                "year"
            ],
            y=region_history[
                "rating_score_mean_t"
            ],
            mode="lines",
            name="Media regional",
            line=dict(
                color="#A0A0A0",
                width=2,
                dash="dash"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                year + 1
            ],
            y=[
                prediction
            ],
            mode="markers",
            name="Predicción",
            marker=dict(
                color="#EF4444",
                size=13
            )
        )
    )

    fig.update_layout(
        title="Evolución histórica del rating soberano",
        template="plotly_white",
        height=530,
        xaxis_title="Año",
        yaxis_title="Rating Score",
        yaxis=dict(
            range=[0, 105]
        ),
        legend=dict(
            orientation="h",
            y=-0.18
        ),
        margin=dict(
            l=45,
            r=25,
            t=55,
            b=65
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def get_model_importance(
    model,
    feature_names,
    top_n=8
):

    importance = pd.DataFrame({
        "variable": feature_names,
        "importance":
        model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(
            top_n
        )
        .sort_values(
            "importance"
        )
    )

    return importance


def get_available_prediction_years(
    panel,
    country_iso
):

    years = (
        panel[
            panel["iso3"]
            == country_iso
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


def get_available_countries(
    panel
):

    countries = (
        panel[
            [
                "iso3",
                "country"
            ]
        ]
        .drop_duplicates()
        .dropna()
        .sort_values(
            "country"
        )
        .reset_index(
            drop=True
        )
    )

    return countries


# perfoles
def get_similar_countries(
    cluster_results,
    country_iso,
    panel,
    model,
    preprocessor,
    features,
    year,
    top_n=3
):

    cluster_frame = None

    for value in cluster_results.values():

        if isinstance(value, pd.DataFrame):

            required_columns = {
                "iso3",
                "country",
                "Cluster",
                "PC1",
                "PC2"
            }

            if required_columns.issubset(value.columns):

                cluster_frame = value.copy()
                break


    if cluster_frame is None:

        return {
            "profile": ".",
            "countries": []
        }


    selected_country = cluster_frame[
        cluster_frame["iso3"] == country_iso
    ].copy()


    if selected_country.empty:

        return {
            "profile": ".",
            "countries": []
        }


    selected_cluster = (
        selected_country["Cluster"].iloc[0]
    )


    if "Perfil de riesgo" in selected_country.columns:

        profile = (
            selected_country[
                "Perfil de riesgo"
            ].iloc[0]
        )

    else:

        profile = "."


    pc1 = selected_country["PC1"].iloc[0]
    pc2 = selected_country["PC2"].iloc[0]


    similar = cluster_frame[
        (cluster_frame["Cluster"] == selected_cluster)
        & (cluster_frame["iso3"] != country_iso)
    ].copy()


    similar["distance"] = np.sqrt(
        (similar["PC1"] - pc1) ** 2
        +
        (similar["PC2"] - pc2) ** 2
    )


    similar = similar.sort_values(
        "distance"
    )


    results = []


    for _, row in similar.iterrows():

        try:

            prediction_result = predict_country(
                panel=panel,
                model=model,
                preprocessor=preprocessor,
                features=features,
                country_iso=row["iso3"],
                year=year
            )

            country_prediction = (
                prediction_result["prediction"]
            )


            results.append({
                "country": row["country"],
                "iso3": row["iso3"],
                "prediction": country_prediction,
                "rating": score_to_rating(
                    country_prediction
                )
            })

        except Exception:

            continue


        if len(results) >= top_n:
            break


    return {
        "profile": profile,
        "countries": results
    }

def get_country_risk_profile(
    data,
    country_iso,
    year,
    top_n=2
):

    dimension_columns = [
        "Macroeconómica",
        "Fiscal",
        "Externa",
        "Institucional",
        "Política y estabilidad",
        "Historial crediticio"
    ]

    risk_dimensions = calculate_risk_dimensions(
        data,
        year
    )

    country_data = risk_dimensions[
        risk_dimensions["iso3"] == country_iso
    ].copy()

    if country_data.empty:

        return {
            "strengths": [],
            "weaknesses": []
        }

    values = (
        country_data[
            dimension_columns
        ]
        .iloc[0]
        .dropna()
        .sort_values()
    )

    strengths = [
        {
            "dimension": dimension,
            "score": float(score)
        }
        for dimension, score
        in values.head(top_n).items()
    ]

    weaknesses = [
        {
            "dimension": dimension,
            "score": float(score)
        }
        for dimension, score
        in values.tail(top_n)
        .sort_values(
            ascending=False
        )
        .items()
    ]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses
    }