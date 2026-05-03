import os
import io
import re
import tempfile
from datetime import datetime
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable
)

TEAL_DARK   = colors.HexColor("#0D6E7A")
TEAL_MID    = colors.HexColor("#1A9BAA")
TEAL_LIGHT  = colors.HexColor("#5BC8D5")
TEAL_PALE   = colors.HexColor("#D6F2F5")
ACCENT_GOLD = colors.HexColor("#E8A020")
ACCENT_RED  = colors.HexColor("#D94040")
ACCENT_GRN  = colors.HexColor("#27AE60")
TEXT_WHITE  = colors.white
TEXT_GRAY   = colors.HexColor("#8EADB5")
TEXT_BODY   = colors.HexColor("#1A2B30")

LOGO_PATH  = os.path.join(os.path.dirname(__file__), "logo.png")
PAGE_W, PAGE_H = A4
MARGIN     = 18 * mm


def make_donut_chart(entity_dict, title="Entity Distribution"):
    clinical = ["Medication", "Sign_symptom", "History", "Lab_value", "Diagnosis"]
    label_colors = {
        "Medication":   "#1A9BAA",
        "Sign_symptom": "#E8A020",
        "History":      "#27AE60",
        "Lab_value":    "#9B59B6",
        "Diagnosis":    "#D94040",
        "Other":        "#8EADB5",
    }
    counts = {}
    for label, items in entity_dict.items():
        if label in clinical:
            counts[label] = len(items)
        else:
            counts["Other"] = counts.get("Other", 0) + len(items)
    if not counts:
        counts = {"No entities": 1}
        label_colors["No entities"] = "#8EADB5"
    labels = list(counts.keys())
    sizes  = list(counts.values())
    clrs   = [label_colors.get(l, "#8EADB5") for l in labels]
    total  = sum(sizes)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="none")
    ax.pie(sizes, colors=clrs, startangle=90,
           wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5))
    ax.text(0, 0, str(total), ha="center", va="center",
            fontsize=18, fontweight="bold", color="#0D6E7A")
    ax.text(0, -0.22, "entities", ha="center", va="center",
            fontsize=8, color="#8EADB5")
    legend_items = [
        mpatches.Patch(color=label_colors.get(l, "#8EADB5"),
                       label=f"{l}  ({counts[l]})")
        for l in labels
    ]
    ax.legend(handles=legend_items, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False)
    ax.set_title(title, fontsize=9, color="#0D6E7A", fontweight="bold", pad=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def make_category_donut(results):
    cat_counts = Counter(r["category"] for r in results)
    cat_colors = {
        "Discharge / General Ward": "#27AE60",
        "ICU / Critical Care":      "#D94040",
        "Outpatient":               "#1A9BAA",
    }
    labels = list(cat_counts.keys())
    sizes  = list(cat_counts.values())
    clrs   = [cat_colors.get(l, "#8EADB5") for l in labels]
    total  = sum(sizes)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="none")
    ax.pie(sizes, colors=clrs, startangle=90,
           wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5))
    ax.text(0, 0, str(total), ha="center", va="center",
            fontsize=18, fontweight="bold", color="#0D6E7A")
    ax.text(0, -0.22, "notes", ha="center", va="center",
            fontsize=8, color="#8EADB5")
    legend_items = [
        mpatches.Patch(color=cat_colors.get(l, "#8EADB5"),
                       label=f"{l}  ({cat_counts[l]})")
        for l in labels
    ]
    ax.legend(handles=legend_items, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False)
    ax.set_title("Notes by Category", fontsize=9,
                 color="#0D6E7A", fontweight="bold", pad=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


class ClinIQPageTemplate:
    def __init__(self, logo_path):
        self.logo_path = logo_path

    def on_page(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(TEAL_DARK)
        canvas.rect(0, h - 14*mm, w, 14*mm, fill=1, stroke=0)
        if os.path.exists(self.logo_path):
            try:
                canvas.drawImage(self.logo_path, MARGIN, h - 12*mm,
                                 width=28*mm, height=9*mm,
                                 preserveAspectRatio=True, mask="auto")
            except Exception:
                pass
        canvas.setFillColor(TEXT_WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawRightString(w - MARGIN, h - 8*mm,
                               "ClinIQ  |  Clinical Decision Support Report")
        canvas.setFillColor(TEAL_DARK)
        canvas.rect(0, 0, w, 10*mm, fill=1, stroke=0)
        canvas.setFillColor(TEXT_WHITE)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, 3.5*mm,
            "ClinIQ - AI-powered clinical decision support. For use by licensed clinicians only.")
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawRightString(w - MARGIN, 3.5*mm,
            f"Page {doc.page}  |  {datetime.now().strftime('%d %b %Y')}")
        canvas.restoreState()


def get_styles():
    styles = {}
    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=36, textColor=TEAL_DARK,
        fontName="Helvetica-Bold", alignment=TA_CENTER, leading=42, spaceAfter=4)
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, textColor=TEAL_MID,
        fontName="Helvetica", alignment=TA_CENTER, leading=18, spaceAfter=2)
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontSize=10, textColor=TEXT_GRAY,
        fontName="Helvetica", alignment=TA_CENTER, leading=14)
    styles["section_title"] = ParagraphStyle(
        "section_title", fontSize=13, textColor=TEAL_DARK,
        fontName="Helvetica-Bold", leading=16, spaceBefore=10, spaceAfter=4)
    styles["card_title"] = ParagraphStyle(
        "card_title", fontSize=10, textColor=TEAL_MID,
        fontName="Helvetica-Bold", leading=13, spaceBefore=2, spaceAfter=2)
    styles["body"] = ParagraphStyle(
        "body", fontSize=9, textColor=TEXT_BODY,
        fontName="Helvetica", leading=13, spaceAfter=4)
    styles["body_white"] = ParagraphStyle(
        "body_white", fontSize=9, textColor=TEXT_WHITE,
        fontName="Helvetica", leading=13, spaceAfter=4)
    styles["note_id"] = ParagraphStyle(
        "note_id", fontSize=11, textColor=TEAL_DARK,
        fontName="Helvetica-Bold", leading=14, spaceBefore=6)
    styles["risk_high"] = ParagraphStyle(
        "risk_high", fontSize=9, textColor=ACCENT_RED,
        fontName="Helvetica-Bold", leading=13)
    styles["explanation"] = ParagraphStyle(
        "explanation", fontSize=8.5, textColor=TEXT_BODY,
        fontName="Helvetica", leading=13, spaceAfter=3)
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer", fontSize=7.5, textColor=TEXT_GRAY,
        fontName="Helvetica", alignment=TA_CENTER, leading=11)
    return styles


def category_color(cat):
    if "ICU" in cat:
        return ACCENT_RED
    elif "Discharge" in cat:
        return ACCENT_GRN
    return TEAL_MID


def generate_pdf(results: list) -> str:
    tmp      = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    out_path = tmp.name
    tmp.close()

    tmpl = ClinIQPageTemplate(LOGO_PATH)
    doc  = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20*mm, bottomMargin=16*mm,
    )

    styles = get_styles()
    story  = []

    # COVER
    story.append(Spacer(1, 28*mm))
    if os.path.exists(LOGO_PATH):
        try:
            story.append(Image(LOGO_PATH, width=70*mm, height=22*mm, hAlign="CENTER"))
        except Exception:
            pass
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Clinical Decision Support", styles["cover_title"]))
    story.append(Paragraph("AI-Powered Patient Note Analysis Report", styles["cover_sub"]))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="60%", thickness=1.5, color=TEAL_LIGHT, hAlign="CENTER"))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%A, %d %B %Y  %H:%M')}",
        styles["cover_meta"]))
    story.append(Paragraph(
        f"Total Notes Analyzed: <b>{len(results)}</b>",
        styles["cover_meta"]))
    story.append(Spacer(1, 10*mm))
    cat_buf = make_category_donut(results)
    story.append(Image(cat_buf, width=110*mm, height=85*mm, hAlign="CENTER"))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "This report is generated by ClinIQ, an AI-powered clinical decision support system. "
        "All findings are intended as decision aids for licensed clinicians and do not constitute "
        "final diagnoses or medical prescriptions.",
        styles["disclaimer"]))
    story.append(PageBreak())

    # OVERVIEW TABLE
    story.append(Paragraph("Report Overview", styles["section_title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL_PALE))
    story.append(Spacer(1, 4*mm))
    table_data = [[
        Paragraph("<b>Note ID</b>", styles["card_title"]),
        Paragraph("<b>Language</b>", styles["card_title"]),
        Paragraph("<b>Category</b>", styles["card_title"]),
        Paragraph("<b>Context</b>", styles["card_title"]),
    ]]
    for r in results:
        table_data.append([
            Paragraph(r["note_id"], styles["body"]),
            Paragraph(r["lang"].upper(), styles["body"]),
            Paragraph(r["category"], styles["body"]),
            Paragraph(r["context_source"], styles["body"]),
        ])
    col_w = [(PAGE_W - 2*MARGIN) / 4] * 4
    tbl = Table(table_data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  TEAL_DARK),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  TEXT_WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [TEAL_PALE, colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, TEAL_LIGHT),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # PER-NOTE PAGES
    for r in results:
        cat   = r["category"]
        c_col = category_color(cat)
        hdr_data = [[
            Paragraph(f"<b>{r['note_id']}</b>", styles["body_white"]),
            Paragraph(cat, styles["body_white"]),
            Paragraph(f"Language: {r['lang'].upper()}", styles["body_white"]),
            Paragraph(f"Source: {r['context_source']}", styles["body_white"]),
        ]]
        hdr_tbl = Table(hdr_data, colWidths=[35*mm, 68*mm, 40*mm, 32*mm])
        hdr_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), c_col),
            ("TEXTCOLOR",     (0, 0), (-1, -1), TEXT_WHITE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(hdr_tbl)
        story.append(Spacer(1, 4*mm))

        summary_p = [
            Paragraph("Summary", styles["card_title"]),
            Paragraph(r["summary"], styles["body"]),
        ]
        donut_buf = make_donut_chart(r["entities"], "Entity Breakdown")
        donut_img = Image(donut_buf, width=85*mm, height=65*mm)
        side_table = Table(
            [[summary_p, donut_img]],
            colWidths=[PAGE_W - 2*MARGIN - 90*mm, 90*mm]
        )
        side_table.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))
        story.append(side_table)
        story.append(Spacer(1, 3*mm))

        story.append(Paragraph("Extracted Medical Entities", styles["card_title"]))
        ent_colors = {
            "Medication":   colors.HexColor("#D6F2F5"),
            "Sign_symptom": colors.HexColor("#FFF3CD"),
            "History":      colors.HexColor("#D5F5E3"),
            "Lab_value":    colors.HexColor("#EBD9F7"),
            "Diagnosis":    colors.HexColor("#FDDEDE"),
        }
        clinical_labels = ["Medication", "Sign_symptom", "History", "Lab_value", "Diagnosis"]
        ent_rows = []
        for lbl in clinical_labels:
            if lbl in r["entities"] and r["entities"][lbl]:
                items = ", ".join(r["entities"][lbl])
                ent_rows.append([
                    Paragraph(f"<b>{lbl}</b>", styles["card_title"]),
                    Paragraph(items, styles["body"]),
                ])
        if ent_rows:
            et = Table(ent_rows, colWidths=[45*mm, PAGE_W - 2*MARGIN - 45*mm])
            ts = TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.3, TEAL_LIGHT),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ])
            for i, row in enumerate(ent_rows):
                lbl_key = clinical_labels[i] if i < len(clinical_labels) else "Other"
                ts.add("BACKGROUND", (0, i), (-1, i),
                       ent_colors.get(lbl_key, TEAL_PALE))
            et.setStyle(ts)
            story.append(et)

        story.append(Spacer(1, 4*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("AI Clinical Decision Support", styles["section_title"]))

        for line in r["explanation"].split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 2))
                continue
            if re.match(r'^#{1,3}\s', line):
                clean = re.sub(r'^#{1,3}\s+', '', line)
                clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean)
                story.append(Paragraph(clean, styles["card_title"]))
            elif "HIGH RISK" in line.upper():
                clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                clean = re.sub(r'^\s*[-*]\s*', '', clean)
                risk_data = [[Paragraph(f"! {clean}", styles["risk_high"])]]
                risk_tbl  = Table(risk_data, colWidths=[PAGE_W - 2*MARGIN])
                risk_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FDDEDE")),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW",     (0, 0), (-1, -1), 1, ACCENT_RED),
                ]))
                story.append(risk_tbl)
                story.append(Spacer(1, 2))
            elif re.match(r'^[-*]\s', line) or re.match(r'^\d+\.\s', line):
                clean = re.sub(r'^[-*]\s*', '- ', line)
                clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean)
                story.append(Paragraph(clean, styles["explanation"]))
            else:
                clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                story.append(Paragraph(clean, styles["explanation"]))

        story.append(PageBreak())

    doc.build(story, onFirstPage=tmpl.on_page, onLaterPages=tmpl.on_page)
    return out_path
