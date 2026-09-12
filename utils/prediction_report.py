from io import BytesIO
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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
    PageBreak,
    KeepTogether
)


def _risk_colors(risk_level):

    colors_map = {
        "Muy bajo": ("#16A34A", "#DCFCE7"),
        "Bajo": ("#15803D", "#E8F5E9"),
        "Medio": ("#B45309", "#FEF3C7"),
        "Alto": ("#EA580C", "#FFEDD5"),
        "Muy alto": ("#DC2626", "#FEE2E2")
    }

    return colors_map.get(
        risk_level,
        ("#64748B", "#F1F5F9")
    )


def _change_color(change):

    if change is None:
        return "#64748B"

    if change > 0:
        return "#16A34A"

    if change < 0:
        return "#DC2626"

    return "#64748B"


def _plotly_image(
    fig,
    width=17.5 * cm,
    height=7.2 * cm
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


def _page_footer(
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

    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.setFont(
        "Helvetica",
        7
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


def create_prediction_report(
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
    similar_results,
    country_risk_profile,
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
        topMargin=1.5 * cm,
        bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B")
    )

    center_style = ParagraphStyle(
        "CenterCustom",
        parent=normal_style,
        alignment=TA_CENTER
    )

    prediction_style = ParagraphStyle(
        "PredictionCustom",
        parent=center_style,
        fontName="Helvetica-Bold",
        fontSize=30,
        leading=34,
        textColor=colors.HexColor("#1E293B")
    )

    rating_style = ParagraphStyle(
        "RatingCustom",
        parent=center_style,
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1E293B")
    )

    story = []
    image_buffers = []

    country_name_safe = escape(
        str(country_name)
    )

    risk_color, risk_background = (
        _risk_colors(
            risk_level
        )
    )

    change_color = _change_color(
        change
    )

    if change is None:
        change_text = "."
    else:
        change_text = f"{change:+.1f} puntos"

    story.append(
        Paragraph(
            f"Informe de predicción de riesgo soberano - "
            f"{country_name_safe}",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Indicadores {reference_year} | "
            f"Predicción {target_year} | "
            f"Código {country_iso}",
            subtitle_style
        )
    )

    prediction_card = Table(
        [
            [
                Paragraph(
                    "Predicción del modelo",
                    section_style
                ),
                Paragraph(
                    "Rating equivalente",
                    section_style
                ),
                Paragraph(
                    "Cambio estimado",
                    section_style
                )
            ],
            [
                Paragraph(
                    f"{prediction:.1f} / 100",
                    prediction_style
                ),
                Paragraph(
                    escape(
                        str(equivalent_rating)
                    ),
                    rating_style
                ),
                Paragraph(
                    (
                        f'<font color="{change_color}">'
                        f"<b>{change_text}</b>"
                        f"</font>"
                    ),
                    rating_style
                )
            ],
            [
                Paragraph(
                    (
                        f'<font color="{risk_color}">'
                        f"<b>Riesgo {risk_level.lower()}</b>"
                        f"</font>"
                    ),
                    center_style
                ),
                Paragraph(
                    "Equivalencia aproximada",
                    small_style
                ),
                Paragraph(
                    (
                        "Respecto al rating medio "
                        f"de {reference_year}"
                    ),
                    small_style
                )
            ]
        ],
        colWidths=[
            6 * cm,
            5 * cm,
            6 * cm
        ]
    )

    prediction_card.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#D7E3F0")
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#E2E8F0")
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
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
                ),
                (
                    "BACKGROUND",
                    (0, 2),
                    (0, 2),
                    colors.HexColor(
                        risk_background
                    )
                )
            ]
        )
    )

    story.append(
        prediction_card
    )

    story.append(
        Spacer(
            1,
            0.4 * cm
        )
    )

    story.append(
        Paragraph(
            "Calificaciones de las agencias",
            section_style
        )
    )
    
    sp_rating = escape(
        str(
            agency_ratings.get(
                "S&P",
                "."
            )
        )
    )
    
    moodys_rating = escape(
        str(
            agency_ratings.get(
                "Moody's",
                "."
            )
        )
    )
    
    fitch_rating = escape(
        str(
            agency_ratings.get(
                "Fitch",
                "."
            )
        )
    )

    agencies_table = Table(
        [
            [
                Paragraph(
                    "<b>S&P</b>",
                    center_style
                ),
                Paragraph(
                    "<b>Moody's</b>",
                    center_style
                ),
                Paragraph(
                    "<b>Fitch</b>",
                    center_style
                )
            ],
            [
                Paragraph(
                    f"<b>{sp_rating}</b>",
                    rating_style
                ),
                Paragraph(
                    f"<b>{moodys_rating}</b>",
                    rating_style
                ),
                Paragraph(
                    f"<b>{fitch_rating}</b>",
                    rating_style
                )
            ]
        ],
        colWidths=[
            5.65 * cm,
            5.65 * cm,
            5.65 * cm
        ]
    )

    agencies_table.setStyle(
        TableStyle(
            [
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#D7E3F0")
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#E2E8F0")
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F8FAFC")
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
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
        agencies_table
    )

    story.append(
        Spacer(
            1,
            0.35 * cm
        )
    )

    interpretation_status = escape(
        str(
            interpretation.get(
                "status",
                ""
            )
        )
    )

    interpretation_text = escape(
        str(
            interpretation.get(
                "text",
                ""
            )
        )
    )

    interpretation_table = Table(
        [
            [
                Paragraph(
                    "<b>Interpretación rápida</b>",
                    normal_style
                )
            ],
            [
                Paragraph(
                    (
                        f"<b>{interpretation_status}</b><br/>"
                        f"{interpretation_text}"
                    ),
                    normal_style
                )
            ]
        ],
        colWidths=[
            16.95 * cm
        ]
    )

    interpretation_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F0F9F4")
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#B7E4C7")
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
            0.35 * cm
        )
    )

    try:

        evolution_image, evolution_buffer = (
            _plotly_image(
                fig_evolution
            )
        )

        image_buffers.append(
            evolution_buffer
        )

        story.append(
            Paragraph(
                "Evolución histórica del rating soberano",
                section_style
            )
        )

        story.append(
            evolution_image
        )

    except Exception:

        story.append(
            Paragraph(
                (
                    "No se pudo incorporar el gráfico "
                    "de evolución al PDF."
                ),
                small_style
            )
        )

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Perfil comparativo",
            section_style
        )
    )

    similar_rows = [
        [
            Paragraph(
                "<b>País similar</b>",
                center_style
            ),
            Paragraph(
                "<b>Predicción</b>",
                center_style
            ),
            Paragraph(
                "<b>Rating</b>",
                center_style
            )
        ]
    ]

    for country in similar_results.get(
        "countries",
        []
    ):

        similar_rows.append(
            [
                Paragraph(
                    escape(
                        str(
                            country[
                                "country"
                            ]
                        )
                    ),
                    normal_style
                ),
                Paragraph(
                    f'{country["prediction"]:.1f} / 100',
                    center_style
                ),
                Paragraph(
                    f'<b>{escape(str(country["rating"]))}</b>',
                    center_style
                )
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
                    colors.HexColor("#F8FAFC")
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#D7E3F0")
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#E2E8F0")
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
        Paragraph(
            (
                "Países con perfil similar - "
                f"perfil {escape(str(similar_results.get('profile', '.')))}"
            ),
            section_style
        )
    )

    story.append(
        similar_table
    )

    story.append(
        Spacer(
            1,
            0.45 * cm
        )
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
                '<font color="#16A34A"><b>Fortalezas relativas</b></font>',
                normal_style
            ),
            Paragraph(
                '<font color="#DC2626"><b>Vulnerabilidades</b></font>',
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

            strength = strengths[i]

            strength_text = (
                f'{escape(str(strength["dimension"]))}: '
                f'<font color="#16A34A">'
                f'<b>{strength["score"]:.1f}</b>'
                f'</font>'
            )

        else:
            strength_text = ""


        if i < len(weaknesses):

            weakness = weaknesses[i]

            weakness_text = (
                f'{escape(str(weakness["dimension"]))}: '
                f'<font color="#DC2626">'
                f'<b>{weakness["score"]:.1f}</b>'
                f'</font>'
            )

        else:
            weakness_text = ""


        profile_rows.append(
            [
                Paragraph(
                    strength_text,
                    normal_style
                ),
                Paragraph(
                    weakness_text,
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
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#D7E3F0")
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#E2E8F0")
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F8FAFC")
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
        profile_table
    )

    story.append(
        Paragraph(
            (
                "Escala 0-100: una menor puntuación "
                "indica menor vulnerabilidad."
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

    model_table = Table(
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
            3.5 * cm,
            3.2 * cm,
            3.2 * cm,
            3.5 * cm,
            3.5 * cm
        ]
    )

    model_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F1F5F9")
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
                    "Helvetica"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
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
                    colors.HexColor("#D7E3F0")
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#E2E8F0")
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
        Paragraph(
            "Información del modelo",
            section_style
        )
    )

    story.append(
        model_table
    )

    story.append(
        Spacer(
            1,
            0.35 * cm
        )
    )

    try:

        importance_image, importance_buffer = (
            _plotly_image(
                fig_importance,
                width=17.5 * cm,
                height=6.7 * cm
            )
        )

        image_buffers.append(
            importance_buffer
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
                (
                    "No se pudo incorporar el gráfico "
                    "de importancia de variables al PDF."
                ),
                small_style
            )
        )

    story.append(
        Spacer(
            1,
            0.25 * cm
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Nota metodológica.</b> "
                "La predicción se genera mediante el modelo "
                "Random Forest definitivo utilizando indicadores "
                f"del año {reference_year} para estimar el rating "
                f"soberano de {target_year}. "
                "Los resultados tienen carácter analítico y no "
                "constituyen una calificación crediticia emitida "
                "por una agencia de rating."
            ),
            small_style
        )
    )

    doc.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()