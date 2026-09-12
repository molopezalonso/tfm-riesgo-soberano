import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler

from utils.risk_dimensions import calculate_risk_dimensions


DIMENSION_COLUMNS = [
    "Macroeconómica",
    "Fiscal",
    "Externa",
    "Institucional",
    "Política y estabilidad",
    "Historial crediticio"
]


DISPLAY_DIMENSIONS = {
    "Macroeconómica": "Macro",
    "Fiscal": "Fiscal",
    "Externa": "Externo",
    "Institucional": "Institucional",
    "Política y estabilidad": "Política",
    "Historial crediticio": "Historial cred."
}


METADATA_COLUMNS = [
    "iso3",
    "iso2",
    "country",
    "region",
    "income_level",
    "year",
    "economic_period"
]


FIXED_EXCLUSIONS = [
    "head_of_government",
    "battle_related_deaths",
    "log_battle_related_deaths",
    "trade_openness_pct_gdp",
    "sovereign_default_dummy",
    "rating_score_mean",
    "rating_year",
    "target_available_t1"
]


RISK_COLORS = [
    "#258BD2",
    "#20B26B",
    "#F2C94C",
    "#F2994A",
    "#EB5757"
]


def get_cluster_candidates(data):

    target_columns = [
        column
        for column in data.columns
        if column.endswith("_t1")
    ]

    excluded = (
        METADATA_COLUMNS
        + FIXED_EXCLUSIONS
        + target_columns
    )

    candidates = [
        column
        for column in data.columns
        if column not in excluded
    ]

    return candidates


def select_cluster_variables(
    data,
    year,
    variables,
    region=None,
    max_missing=40
):

    selected_data = (
        data[
            data["year"] == year
        ]
        .copy()
    )

    if region is not None:

        selected_data = (
            selected_data[
                selected_data["region"] == region
            ]
            .copy()
        )

    selected_variables = []
    excluded_rows = []

    for variable in variables:

        if variable not in selected_data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            selected_data[variable]
        ):

            excluded_rows.append({
                "Variable": variable,
                "Motivo": "No numérica"
            })

            continue

        missing_pct = (
            selected_data[variable]
            .isna()
            .mean()
            * 100
        )

        if missing_pct > max_missing:

            excluded_rows.append({
                "Variable": variable,
                "Motivo": f"Missing {missing_pct:.1f}%"
            })

            continue

        unique_values = (
            selected_data[variable]
            .nunique(
                dropna=True
            )
        )

        if unique_values <= 1:

            excluded_rows.append({
                "Variable": variable,
                "Motivo": "Sin variabilidad"
            })

            continue

        selected_variables.append(
            variable
        )

    excluded_table = pd.DataFrame(
        excluded_rows
    )

    return (
        selected_data,
        selected_variables,
        excluded_table
    )


def prepare_cluster_data(
    selected_data,
    selected_variables,
    max_country_missing=40
):

    id_columns = [
        column
        for column in [
            "iso3",
            "country",
            "region"
        ]
        if column in selected_data.columns
    ]

    cluster_data = (
        selected_data[
            id_columns
            + selected_variables
        ]
        .copy()
    )

    country_missing = (
        cluster_data[
            selected_variables
        ]
        .isna()
        .mean(axis=1)
        * 100
    )

    cluster_data = (
        cluster_data[
            country_missing
            <= max_country_missing
        ]
        .copy()
    )

    X = (
        cluster_data[
            selected_variables
        ]
        .copy()
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = (
        imputer.fit_transform(
            X
        )
    )

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            X_imputed
        )
    )

    return (
        cluster_data,
        X_scaled,
        imputer,
        scaler
    )


def select_pca_components(
    X_scaled,
    min_variance=0.60,
    min_component_variance=0.04,
    max_components=10
):

    pca_full = PCA()

    pca_full.fit(
        X_scaled
    )

    explained_variance = (
        pca_full
        .explained_variance_ratio_
    )

    cumulative_variance = (
        np.cumsum(
            explained_variance
        )
    )

    n_selected = None

    for i in range(
        len(explained_variance)
    ):

        n_components = i + 1

        enough_variance = (
            cumulative_variance[i]
            >= min_variance
        )

        small_component = (
            explained_variance[i]
            < min_component_variance
        )

        if (
            enough_variance
            and small_component
        ):

            n_selected = n_components
            break

    if n_selected is None:

        positions = np.where(
            cumulative_variance
            >= min_variance
        )[0]

        if len(positions) > 0:

            n_selected = (
                positions[0]
                + 1
            )

        else:

            n_selected = (
                len(explained_variance)
            )

    n_selected = min(
        n_selected,
        max_components,
        len(explained_variance)
    )

    return {
        "n_components":
            n_selected,

        "explained_variance":
            cumulative_variance[
                n_selected - 1
            ],

        "individual_variance":
            explained_variance[
                n_selected - 1
            ],

        "explained_variance_all":
            explained_variance,

        "cumulative_variance":
            cumulative_variance
    }


def apply_pca(
    X_scaled
):

    selection = (
        select_pca_components(
            X_scaled
        )
    )

    n_components = (
        selection[
            "n_components"
        ]
    )

    pca = PCA(
        n_components=n_components
    )

    X_pca = (
        pca.fit_transform(
            X_scaled
        )
    )

    return (
        X_pca,
        pca,
        selection
    )


def select_number_clusters(
    X,
    k_min=3,
    k_max=6,
    min_cluster_pct=0.02,
    min_cluster_abs=3
):

    results = []

    n_samples = len(X)

    if n_samples < 3:
        raise ValueError(
            "Se necesitan al menos 3 países "
            "para realizar el análisis de clústeres."
        )

    max_possible_k = min(
        k_max,
        n_samples - 1
    )

    min_possible_k = min(
        k_min,
        max_possible_k
    )

    min_possible_k = max(
        2,
        min_possible_k
    )

    for k in range(
        min_possible_k,
        max_possible_k + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X)

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
        )

        results.append({
            "K": k,
            "Inercia": model.inertia_,
            "Silhouette": silhouette_score(
                X,
                labels
            ),
            "Davies-Bouldin": davies_bouldin_score(
                X,
                labels
            ),
            "Calinski-Harabasz": calinski_harabasz_score(
                X,
                labels
            ),
            "Cluster mínimo": cluster_sizes.min()
        })

    results = pd.DataFrame(results)

    min_size = max(
        min_cluster_abs,
        int(
            np.ceil(
                n_samples
                * min_cluster_pct
            )
        )
    )

    valid_results = (
        results[
            results["Cluster mínimo"]
            >= min_size
        ]
        .copy()
    )

    if valid_results.empty:
        valid_results = results.copy()

    best_k = int(
        valid_results.loc[
            valid_results["Silhouette"].idxmax(),
            "K"
        ]
    )

    return best_k, results


def get_risk_names(k):

    if k == 3:

        return [
            "Bajo",
            "Medio",
            "Alto"
        ]

    if k == 4:

        return [
            "Bajo",
            "Medio",
            "Alto",
            "Muy alto"
        ]

    if k == 5:

        return [
            "Muy bajo",
            "Bajo",
            "Medio",
            "Alto",
            "Muy alto"
        ]

    return [
        "Muy bajo",
        "Bajo",
        "Medio-bajo",
        "Medio-alto",
        "Alto",
        "Muy alto"
    ]


def get_risk_colors(k):

    if k == 3:

        return [
            "#258BD2",
            "#F2C94C",
            "#EB5757"
        ]

    if k == 4:

        return [
            "#258BD2",
            "#20B26B",
            "#F2994A",
            "#EB5757"
        ]

    if k == 5:

        return RISK_COLORS

    return [
        "#258BD2",
        "#33A6B8",
        "#20B26B",
        "#F2C94C",
        "#F2994A",
        "#EB5757"
    ]


def assign_risk_profiles(
    cluster_data
):

    cluster_risk = (
        cluster_data
        .groupby(
            "Cluster"
        )[
            "Risk_Score"
        ]
        .mean()
        .sort_values()
    )

    ordered_clusters = (
        cluster_risk
        .index
        .tolist()
    )

    k = len(
        ordered_clusters
    )

    risk_names = (
        get_risk_names(k)
    )

    risk_colors = (
        get_risk_colors(k)
    )

    risk_name_map = dict(
        zip(
            ordered_clusters,
            risk_names
        )
    )

    color_map = dict(
        zip(
            ordered_clusters,
            risk_colors
        )
    )

    cluster_data = (
        cluster_data.copy()
    )

    cluster_data[
        "Perfil de riesgo"
    ] = (
        cluster_data[
            "Cluster"
        ]
        .map(
            risk_name_map
        )
    )

    cluster_data[
        "Cluster_display"
    ] = (
        cluster_data[
            "Cluster"
        ]
        .apply(
            lambda x:
            f"Clúster {x} · "
            f"{risk_name_map[x]}"
        )
    )

    display_colors = {
        f"Clúster {cluster} · "
        f"{risk_name_map[cluster]}":
            color_map[cluster]

        for cluster
        in ordered_clusters
    }

    return (
        cluster_data,
        cluster_risk,
        risk_name_map,
        color_map,
        display_colors
    )


def add_pca_labels(
    cluster_data,
    country_reference="Spain",
    n_groups=20
):

    result = (
        cluster_data.copy()
    )

    n_groups = min(
        n_groups,
        len(result)
    )

    result[
        "pc1_group"
    ] = pd.qcut(
        result["PC1"],
        q=n_groups,
        duplicates="drop"
    )

    labels = []

    for group in (
        result[
            "pc1_group"
        ]
        .dropna()
        .unique()
    ):

        group_data = (
            result[
                result[
                    "pc1_group"
                ] == group
            ]
        )

        selected = pd.concat([
            group_data.nlargest(
                1,
                "PC2"
            ),
            group_data.nsmallest(
                1,
                "PC2"
            )
        ])

        labels.extend(
            selected[
                "country"
            ]
            .tolist()
        )

    extremes = [
        result.nsmallest(
            1,
            "PC1"
        )["country"].iloc[0],

        result.nlargest(
            1,
            "PC1"
        )["country"].iloc[0],

        result.nsmallest(
            1,
            "PC2"
        )["country"].iloc[0],

        result.nlargest(
            1,
            "PC2"
        )["country"].iloc[0]
    ]

    labels.extend(
        extremes
    )

    if (
        country_reference
        in result["country"].values
    ):

        labels.append(
            country_reference
        )

    labels = list(
        dict.fromkeys(
            labels
        )
    )

    result[
        "label"
    ] = np.where(
        result["country"]
        .isin(labels),
        result["country"],
        ""
    )

    result[
        "pc1_group"
    ] = (
        result[
            "pc1_group"
        ]
        .astype(str)
    )

    return result


def get_cluster_drivers(row):

    values = (
        row[
            DIMENSION_COLUMNS
        ]
        .sort_values(
            ascending=False
        )
    )

    return (
        values.index[0]
        + " · "
        + values.index[1]
    )


def get_cluster_strengths(row):

    values = (
        row[
            DIMENSION_COLUMNS
        ]
        .sort_values()
    )

    return (
        values.index[0]
        + " y "
        + values.index[1]
    )


def create_cluster_summary(
    cluster_data,
    risk_dimensions
):

    cluster_dimensions = (
        cluster_data[
            [
                "iso3",
                "country",
                "Cluster",
                "Perfil de riesgo",
                "Risk_Score"
            ]
        ]
        .merge(
            risk_dimensions[
                [
                    "iso3"
                ]
                + DIMENSION_COLUMNS
            ],
            on="iso3",
            how="left"
        )
    )

    summary = (
        cluster_dimensions
        .groupby(
            [
                "Cluster",
                "Perfil de riesgo"
            ],
            as_index=False
        )
        .agg(
            {
                "country":
                    "count",

                "Risk_Score":
                    "mean",

                **{
                    dimension:
                        "mean"

                    for dimension
                    in DIMENSION_COLUMNS
                }
            }
        )
    )

    summary = (
        summary
        .rename(
            columns={
                "country":
                    "Países",

                "Risk_Score":
                    "Risk Score medio"
            }
        )
        .sort_values(
            "Risk Score medio"
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "Rasgos dominantes"
    ] = (
        summary.apply(
            get_cluster_drivers,
            axis=1
        )
    )

    return summary

def create_cluster_table_html(
    cluster_data,
    summary,
    color_map,
    n_examples=4
):

    # Países de ejemplo
    cluster_examples = (
        cluster_data
        .groupby("Cluster")["country"]
        .apply(
            lambda x: ", ".join(
                sorted(x)[:n_examples]
            )
        )
        .to_dict()
    )

    html = """
    <div style="
        background:white;
        border:1px solid #D8E0EA;
        border-radius:14px;
        overflow:hidden;
        width:100%;
    ">

    <table style="
        width:100%;
        border-collapse:collapse;
        background:white;
        font-size:14px;
    ">

    <thead>
        <tr style="
            background:#F8FAFC;
            border-bottom:1px solid #CBD5E1;
        ">
            <th style="padding:14px;text-align:left;">
                Clúster
            </th>

            <th style="padding:14px;text-align:center;">
                Perfil
            </th>

            <th style="padding:14px;text-align:center;">
                Países (ejemplos)
            </th>

            <th style="padding:14px;text-align:center;">
                # Países
            </th>

            <th style="padding:14px;text-align:center;">
                Risk Score medio
            </th>

            <th style="padding:14px;text-align:center;">
                Rasgos dominantes
            </th>
        </tr>
    </thead>

    <tbody>
    """

    for _, row in summary.iterrows():

        cluster = int(
            row["Cluster"]
        )

        color = (
            color_map[cluster]
        )

        countries = (
            cluster_examples.get(
                cluster,
                ""
            )
        )

        html += f"""
        <tr style="
            border-bottom:1px solid #E2E8F0;
        ">

            <td style="
                padding:18px 14px;
                text-align:center;
                font-weight:700;
                border-bottom:3px solid {color};
                white-space:nowrap;
            ">
                Clúster {cluster}
            </td>

            <td style="
                padding:18px 14px;
                text-align:center;
                font-weight:600;
            ">
                {row["Perfil de riesgo"]}
            </td>

            <td style="
                padding:18px 14px;
                text-align:center;
                color:#475569;
                line-height:1.5;
            ">
                {countries}
            </td>

            <td style="
                padding:18px 14px;
                text-align:center;
            ">
                {int(row["Países"])}
            </td>

            <td style="
                padding:18px 14px;
                text-align:center;
                font-size:16px;
                font-weight:700;
                color:{color};
            ">
                {row["Risk Score medio"]:.1f}
            </td>

            <td style="
                padding:18px 14px;
                text-align:center;
                color:#334155;
                line-height:1.5;
            ">
                {row["Rasgos dominantes"]}
            </td>

        </tr>
        """

    html += """
    </tbody>
    </table>
    </div>
    """

    return html



def create_quick_findings(
    summary
):

    findings = []

    for _, row in (
        summary.iterrows()
    ):

        cluster = int(
            row["Cluster"]
        )

        profile = (
            row[
                "Perfil de riesgo"
            ]
        )

        vulnerabilities = (
            get_cluster_drivers(
                row
            )
            .replace(
                " · ",
                " y "
            )
            .lower()
        )

        strengths = (
            get_cluster_strengths(
                row
            )
            .lower()
        )

        if profile in [
            "Muy bajo",
            "Bajo"
        ]:

            text = (
                f"El clúster {cluster} presenta "
                f"un perfil de riesgo "
                f"{profile.lower()}, destacando "
                f"su fortaleza en {strengths}."
            )

        elif (
            "Medio"
            in profile
        ):

            text = (
                f"El clúster {cluster} presenta "
                f"un riesgo intermedio, con "
                f"mayor vulnerabilidad en "
                f"{vulnerabilities}."
            )

        else:

            text = (
                f"El clúster {cluster} presenta "
                f"un perfil de riesgo "
                f"{profile.lower()}, principalmente "
                f"asociado a {vulnerabilities}."
            )

        findings.append(
            text
        )

    return findings


def plot_cluster_pca(
    cluster_data,
    display_colors
):

    fig = px.scatter(
        cluster_data,
        x="PC1",
        y="PC2",
        color="Cluster_display",
        text="label",
        hover_name="country",
        hover_data={
            "iso3": True,
            "region": True,
            "Risk_Score": ":.1f",
            "PC1": ":.2f",
            "PC2": ":.2f",
            "Cluster_display": False,
            "pc1_group": False,
            "label": False
        },
        color_discrete_map=
            display_colors,
        title=(
            "Análisis de clústeres "
            "de países"
        )
    )

    fig.update_traces(
        marker=dict(
            size=9
        ),
        textposition=
            "top center",
        textfont=dict(
            size=9,
            color="#AAB2BD"
        )
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Perfil de riesgo"
    )

    return fig


def plot_cluster_sizes(
    summary,
    color_map
):

    fig = go.Figure()

    for _, row in (
        summary.iterrows()
    ):

        cluster = int(
            row["Cluster"]
        )

        label = (
            f"Clúster {cluster} · "
            f"{row['Perfil de riesgo']}"
        )

        fig.add_trace(
            go.Bar(
                x=[
                    row["Países"]
                ],
                y=[
                    label
                ],
                orientation="h",
                marker_color=
                    color_map[
                        cluster
                    ],
                text=[
                    int(
                        row["Países"]
                    )
                ],
                textposition=
                    "outside",
                showlegend=False
            )
        )

    fig.update_layout(
        title="Tamaño de los clústeres",
        xaxis_title="Número de países",
        yaxis_title="",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=120,
            r=40,
            t=60,
            b=50
        )
    )
    
    fig.update_yaxes(
        autorange="reversed"
    )
    
    return fig


def plot_cluster_heatmap(
    summary
):

    heatmap_data = (
        summary
        .set_index(
            "Perfil de riesgo"
        )[
            DIMENSION_COLUMNS
        ]
        .rename(
            columns=
                DISPLAY_DIMENSIONS
        )
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            zmin=0,
            zmax=100,

            colorscale=[
                [0.00, "#CFE3F4"],
                [0.25, "#D5E8DE"],
                [0.50, "#EEEBCB"],
                [0.75, "#EFD7C7"],
                [1.00, "#E7BFC1"]
            ],

            text=np.round(
                heatmap_data.values
            ).astype(int),

            texttemplate=
                "<b>%{text}</b>",

            textfont=dict(
                size=15,
                color="#1E293B"
            ),

            xgap=3,
            ygap=3,

            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{z:.1f}"
                "<extra></extra>"
            ),

            colorbar=dict(
                title="Riesgo",
                thickness=12,
                len=0.75,
                outlinewidth=0
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Perfil de riesgo "
                "por dimensiones"
            ),
            x=0.02,
            xanchor="left",
            font=dict(
                size=20,
                color="#1E293B"
            )
        ),

        height=430,

        margin=dict(
            l=90,
            r=80,
            t=70,
            b=55
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",

        xaxis=dict(
            title="",
            tickangle=0
        ),

        yaxis=dict(
            title="",
            autorange="reversed"
        )
    )

    return fig


def run_cluster_analysis(
    data,
    ratings,
    year,
    region=None,
    selected_k=None
):

    candidates = (
        get_cluster_candidates(
            data
        )
    )

    (
        selected_data,
        selected_variables,
        excluded_variables
    ) = select_cluster_variables(
        data=data,
        year=year,
        variables=candidates,
        region=region
    )

    (
        cluster_sample,
        X_scaled,
        imputer,
        scaler
    ) = prepare_cluster_data(
        selected_data,
        selected_variables
    )

    (
        X_pca,
        pca,
        pca_selection
    ) = apply_pca(
        X_scaled
    )

    if selected_k is None:

        best_k, evaluation = (
            select_number_clusters(
                X_pca
            )
        )

    else:

        best_k = int(
            selected_k
        )

        if best_k > len(X_pca):

            raise ValueError(
                f"No hay suficientes países "
                f"para crear {best_k} clústeres."
            )

    evaluation = pd.DataFrame()

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )

    labels = (
        kmeans.fit_predict(
            X_pca
        )
    )

    cluster_data = (
        cluster_sample[
            [
                "iso3",
                "country",
                "region"
            ]
        ]
        .copy()
    )

    cluster_data["PC1"] = (
        X_pca[:, 0]
    )

    cluster_data["PC2"] = (
        X_pca[:, 1]
    )

    cluster_data["Cluster"] = (
        labels + 1
    )

    ratings_year = (
        ratings[
            ratings["year"] == year
        ][
            [
                "iso3",
                "rating_score_mean"
            ]
        ]
        .copy()
    )

    ratings_year[
        "Risk_Score"
    ] = (
        100
        - ratings_year[
            "rating_score_mean"
        ]
    )

    cluster_data = (
        cluster_data
        .merge(
            ratings_year[
                [
                    "iso3",
                    "Risk_Score"
                ]
            ],
            on="iso3",
            how="left"
        )
    )

    (
        cluster_data,
        cluster_risk,
        risk_name_map,
        color_map,
        display_colors
    ) = assign_risk_profiles(
        cluster_data
    )

    cluster_data = (
        add_pca_labels(
            cluster_data
        )
    )

    risk_dimensions = (
        calculate_risk_dimensions(
            data,
            year
        )
    )

    summary = (
        create_cluster_summary(
            cluster_data,
            risk_dimensions
        )
    )

    table_html = (
        create_cluster_table_html(
            cluster_data,
            summary,
            color_map
        )
    )

    findings = (
        create_quick_findings(
            summary
        )
    )

    fig_pca = (
        plot_cluster_pca(
            cluster_data,
            display_colors
        )
    )

    fig_sizes = (
        plot_cluster_sizes(
            summary,
            color_map
        )
    )

    fig_heatmap = (
        plot_cluster_heatmap(
            summary
        )
    )
    table_html = create_cluster_table_html(
        cluster_data,
        summary,
        color_map
    )

    return {
        "cluster_data": cluster_data,
        "summary": summary,
        "findings": findings,
        "fig_pca": fig_pca,
        "fig_sizes": fig_sizes,
        "fig_heatmap": fig_heatmap,
        "table_html": table_html,
    
        "variables": selected_variables,
        "excluded_variables": excluded_variables,
        "countries_initial": len(selected_data),
        "countries_final": len(cluster_sample),
        "n_components": pca_selection["n_components"],
        "pca_variance": pca_selection["explained_variance"],
        "best_k": best_k,
        "cluster_evaluation": evaluation,
        "pca": pca,
        "kmeans": kmeans,
        "imputer": imputer,
        "scaler": scaler
}