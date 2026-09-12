from io import BytesIO
from html import escape

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


def _safe(value):

    if value is None:
        return "."

    return escape(
        str(value)
    )


def _plotly_image(
    fig,
    width=17.4 * cm,
    height=7 * cm
):

    image_bytes = fig.to_image(
        format="png",
        width=1200,
        height=500,
        scale=2
    )

    buffer = BytesIO(
        image_bytes
    )

    image = Image(
        buffer,
        width=width,
        height=height
    )

    return image, buffer


def _footer(
    canvas,
    doc
):

    canvas.saveState()

    width, _ = A4

    canvas.setStrokeColor(
        colors.HexColor("#D7E3F0")
    )

    canvas.line(
        1.5 * cm,
        1.25 * cm,
        width - 1.5 * cm,
        1.25 * cm
    )

    canvas.setFont(
        "Helvetica",
        7
    )

    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.drawString(
        1.5 * cm,
        0.85 * cm,
        "Herramienta de análisis de riesgo soberano"
    )

    canvas.drawRightString(
        width - 1.5 * cm,
        0.85 * cm,
        f"Página {doc.page}"
    )

    canvas.restoreState()


def _format_indicator(
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

    return f"{value:.2f}"


def get_cluster_report_info(
    cluster_results,
    country_iso
):

    cluster_data = None

    for value in cluster_results.values():

        if isinstance(
            value,
            pd.DataFrame
        ):

            required = {
                "iso3",
                "Cluster"
            }

            if required.issubset(
                value.columns
            ):

                cluster_data = value.copy()
                break


    if cluster_data is None:

        return {
            "cluster": ".",
            "profile": ".",
            "size": 0
        }


    country_row = cluster_data[
        cluster_data["iso3"]
        == country_iso
    ]


    if country_row.empty:

        return {
            "cluster": ".",
            "profile": ".",
            "size": 0
        }


    cluster_number = (
        country_row[
            "Cluster"
        ]
        .iloc[0]
    )


    cluster_size = len(
        cluster_data[
            cluster_data["Cluster"]
            == cluster_number
        ]
    )


    if (
        "Perfil de riesgo"
        in country_row.columns
    ):

        profile = (
            country_row[
                "Perfil de riesgo"
            ]
            .iloc[0]
        )

    else:

        profile = "."


    return {
        "cluster": cluster_number,
        "profile": profile,
        "size": cluster_size
    }


def create_full_country_report(
    country_name,
    country_iso,
    reference_year,
    target_year,
    prediction,
    equivalent_rating,
    risk_level,
    current_rating,
    change,
    agency_ratings,
    interpretation,
    key_indicators,
    alert_results,
    similar_results,
    country_risk_profile,
    cluster_results,
    fig_evolution,
    fig_importance,
    n_features,
    mae_test=4.988,
    r2_test=0.928
):

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.7 * cm
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor(
            "#1E293B"
        ),
        spaceAfter=5
    )


    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(
            "#64748B"
        ),
        spaceAfter=12
    )


    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(
            "#1E293B"
        ),
        spaceBefore=5,
        spaceAfter=7
    )


    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor(
            "#334155"
        )
    )


    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=10,
        textColor=colors.HexColor(
            "#64748B"
        )
    )


    center_style = ParagraphStyle(
        "CenterCustom",
        parent=normal_style,
        alignment=1
    )


    big_style = ParagraphStyle(
        "BigCustom",
        parent=center_style,
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=27,
        textColor=colors.HexColor(
            "#1E293B"
        )
    )


    story = []
    image_buffers = []


    # Información del clúster
    cluster_info = get_cluster_report_info(
        cluster_results,
        country_iso
    )


    # Página 1
    story.append(
        Paragraph(
            (
                "Informe de riesgo soberano - "
                f"{_safe(country_name)}"
            ),
            title_style
        )
    )


    story.append(
        Paragraph(
            (
                f"Indicadores {reference_year} | "
                f"Predicción {target_year} | "
                f"Código {country_iso}"
            ),
            subtitle_style
        )
    )


    if change is None:
        change_text = "."
    else:
        change_text = (
            f"{change:+.1f} puntos"
        )


    summary_table = Table(
        [
            [
                Paragraph(
                    "<b>Predicción</b>",
                    center_style
                ),
                Paragraph(
                    "<b>Rating estimado</b>",
                    center_style
                ),
                Paragraph(
                    "<b>Nivel de riesgo</b>",
                    center_style
                ),
                Paragraph(
                    "<b>Cambio</b>",
                    center_style
                )
            ],
            [
                Paragraph(
                    f"{prediction:.1f}/100",
                    big_style
                ),
                Paragraph(
                    _safe(
                        equivalent_rating
                    ),
                    big_style
                ),
                Paragraph(
                    _safe(
                        risk_level
                    ),
                    big_style
                ),
                Paragraph(
                    change_text,
                    big_style
                )
            ]
        ],
        colWidths=[
            4.25 * cm,
            4.25 * cm,
            4.25 * cm,
            4.25 * cm
        ]
    )


    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F8FAFC"
                    )
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                )
            ]
        )
    )


    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            0.35 * cm
        )
    )


    story.append(
        Paragraph(
            "Calificaciones de las agencias",
            section_style
        )
    )


    agency_table = Table(
        [
            [
                "S&P",
                "Moody's",
                "Fitch"
            ],
            [
                agency_ratings.get(
                    "S&P",
                    "."
                ),
                agency_ratings.get(
                    "Moody's",
                    "."
                ),
                agency_ratings.get(
                    "Fitch",
                    "."
                )
            ]
        ],
        colWidths=[
            5.65 * cm,
            5.65 * cm,
            5.65 * cm
        ]
    )


    agency_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F8FAFC"
                    )
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    10
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )


    story.append(
        agency_table
    )

    story.append(
        Spacer(
            1,
            0.3 * cm
        )
    )


    interpretation_title = (
        interpretation.get(
            "status",
            interpretation.get(
                "title",
                "Interpretación"
            )
        )
    )


    interpretation_text = (
        interpretation.get(
            "text",
            interpretation.get(
                "message",
                ""
            )
        )
    )


    interpretation_table = Table(
        [
            [
                Paragraph(
                    (
                        f"<b>{_safe(interpretation_title)}</b>"
                        f"<br/>{_safe(interpretation_text)}"
                    ),
                    normal_style
                )
            ]
        ],
        colWidths=[
            17 * cm
        ]
    )


    interpretation_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#F0F9F4"
                    )
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#B7E4C7"
                    )
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )


    story.append(
        interpretation_table
    )

    story.append(
        Spacer(
            1,
            0.3 * cm
        )
    )


    try:

        evolution_image, buffer = (
            _plotly_image(
                fig_evolution
            )
        )

        image_buffers.append(
            buffer
        )

        story.append(
            Paragraph(
                "Evolución histórica",
                section_style
            )
        )

        story.append(
            evolution_image
        )

    except Exception:

        story.append(
            Paragraph(
                "No se pudo incorporar el gráfico de evolución.",
                small_style
            )
        )


    # Página 2
    story.append(
        PageBreak()
    )


    story.append(
        Paragraph(
            "Indicadores principales",
            section_style
        )
    )


    indicator_rows = []


    for indicator in key_indicators:

        indicator_rows.append(
            [
                Paragraph(
                    f"<b>{_safe(indicator['label'])}</b>",
                    normal_style
                ),
                Paragraph(
                    _format_indicator(
                        indicator["variable"],
                        indicator["value"]
                    ),
                    center_style
                )
            ]
        )


    indicator_table = Table(
        indicator_rows,
        colWidths=[
            12.5 * cm,
            4.5 * cm
        ]
    )


    indicator_table.setStyle(
        TableStyle(
            [
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )


    story.append(
        indicator_table
    )

    story.append(
        Spacer(
            1,
            0.35 * cm
        )
    )


    # Alertas
    story.append(
        Paragraph(
            "Alertas detectadas",
            section_style
        )
    )


    alerts = alert_results.get(
        "alerts",
        []
    )


    if len(alerts) == 0:

        story.append(
            Paragraph(
                (
                    "<b>Sin alertas relevantes.</b> "
                    "No se detectan señales de riesgo "
                    "relevantes para el año seleccionado."
                ),
                normal_style
            )
        )

    else:

        alert_rows = [
            [
                "Nivel",
                "Alerta",
                "Valor"
            ]
        ]


        for alert in alerts:

            alert_rows.append(
                [
                    alert["severity"],
                    Paragraph(
                        (
                            f"<b>{_safe(alert['title'])}</b>"
                            f"<br/>{_safe(alert['message'])}"
                        ),
                        normal_style
                    ),
                    alert["value"]
                ]
            )


        alert_table = Table(
            alert_rows,
            colWidths=[
                2.4 * cm,
                11.5 * cm,
                3.1 * cm
            ]
        )


        alert_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#F8FAFC"
                        )
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.7,
                        colors.HexColor(
                            "#D7E3F0"
                        )
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.HexColor(
                            "#E2E8F0"
                        )
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )


        story.append(
            alert_table
        )


    story.append(
        Spacer(
            1,
            0.4 * cm
        )
    )


    # Clúster
    story.append(
        Paragraph(
            "Posicionamiento por clúster",
            section_style
        )
    )


    cluster_table = Table(
        [
            [
                "Clúster",
                "Perfil de riesgo",
                "Países en el clúster"
            ],
            [
                str(
                    cluster_info["cluster"]
                ),
                str(
                    cluster_info["profile"]
                ),
                str(
                    cluster_info["size"]
                )
            ]
        ],
        colWidths=[
            5.5 * cm,
            6 * cm,
            5.5 * cm
        ]
    )


    cluster_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F8FAFC"
                    )
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )


    story.append(
        cluster_table
    )


    story.append(
        Spacer(
            1,
            0.25 * cm
        )
    )


    similar_countries = (
        similar_results.get(
            "countries",
            []
        )
    )


    if len(similar_countries) > 0:

        similar_rows = [
            [
                "País comparable",
                "Predicción",
                "Rating"
            ]
        ]


        for country in similar_countries:

            similar_rows.append(
                [
                    country["country"],
                    f'{country["prediction"]:.1f}/100',
                    country["rating"]
                ]
            )


        similar_table = Table(
            similar_rows,
            colWidths=[
                8.5 * cm,
                4.5 * cm,
                4 * cm
            ]
        )


        similar_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#F8FAFC"
                        )
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.7,
                        colors.HexColor(
                            "#D7E3F0"
                        )
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.HexColor(
                            "#E2E8F0"
                        )
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )


        story.append(
            Paragraph(
                "Países con perfil similar",
                section_style
            )
        )

        story.append(
            similar_table
        )


    # Página 3
    story.append(
        PageBreak()
    )


    story.append(
        Paragraph(
            "Fortalezas y vulnerabilidades",
            section_style
        )
    )


    strengths = (
        country_risk_profile.get(
            "strengths",
            []
        )
    )

    weaknesses = (
        country_risk_profile.get(
            "weaknesses",
            []
        )
    )


    profile_rows = [
        [
            Paragraph(
                "<b>Fortalezas relativas</b>",
                normal_style
            ),
            Paragraph(
                "<b>Vulnerabilidades</b>",
                normal_style
            )
        ]
    ]


    max_rows = max(
        len(strengths),
        len(weaknesses),
        1
    )


    for i in range(
        max_rows
    ):

        if i < len(strengths):

            strength_text = (
                f"{strengths[i]['dimension']}: "
                f"{strengths[i]['score']:.1f}"
            )

        else:
            strength_text = ""


        if i < len(weaknesses):

            weakness_text = (
                f"{weaknesses[i]['dimension']}: "
                f"{weaknesses[i]['score']:.1f}"
            )

        else:
            weakness_text = ""


        profile_rows.append(
            [
                Paragraph(
                    _safe(
                        strength_text
                    ),
                    normal_style
                ),
                Paragraph(
                    _safe(
                        weakness_text
                    ),
                    normal_style
                )
            ]
        )


    profile_table = Table(
        profile_rows,
        colWidths=[
            8.5 * cm,
            8.5 * cm
        ]
    )


    profile_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F8FAFC"
                    )
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )


    story.append(
        profile_table
    )


    story.append(
        Paragraph(
            (
                "Escala 0-100: una menor puntuación "
                "indica menor vulnerabilidad relativa."
            ),
            small_style
        )
    )


    story.append(
        Spacer(
            1,
            0.4 * cm
        )
    )


    try:

        importance_image, buffer = (
            _plotly_image(
                fig_importance,
                height=6.8 * cm
            )
        )

        image_buffers.append(
            buffer
        )

        story.append(
            Paragraph(
                "Principales variables del modelo",
                section_style
            )
        )

        story.append(
            importance_image
        )

    except Exception:

        story.append(
            Paragraph(
                "No se pudo incorporar el gráfico de importancia.",
                small_style
            )
        )


    story.append(
        Spacer(
            1,
            0.3 * cm
        )
    )


    story.append(
        Paragraph(
            "Ficha metodológica",
            section_style
        )
    )


    methodology_table = Table(
        [
            [
                "Modelo",
                "Variables",
                "Horizonte",
                "MAE test",
                "R2 test"
            ],
            [
                "Random Forest",
                str(n_features),
                "Rating t+1",
                f"{mae_test:.3f}",
                f"{r2_test:.3f}"
            ]
        ],
        colWidths=[
            3.6 * cm,
            3.2 * cm,
            3.2 * cm,
            3.5 * cm,
            3.5 * cm
        ]
    )


    methodology_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F1F5F9"
                    )
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(
                        "#D7E3F0"
                    )
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor(
                        "#E2E8F0"
                    )
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )


    story.append(
        methodology_table
    )


    story.append(
        Spacer(
            1,
            0.3 * cm
        )
    )


    story.append(
        Paragraph(
            (
                "<b>Nota metodológica.</b> "
                "La predicción utiliza indicadores del año "
                f"{reference_year} para estimar el rating soberano "
                f"de {target_year}. "
                "El posicionamiento por clúster se obtiene a partir "
                "de las características económicas, fiscales, externas, "
                "institucionales, políticas y crediticias del país. "
                "Las dimensiones de riesgo se expresan en una escala "
                "relativa de 0 a 100. "
                "Los resultados tienen carácter analítico y no "
                "constituyen una calificación crediticia emitida "
                "por una agencia de rating."
            ),
            small_style
        )
    )


    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer
    )


    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()