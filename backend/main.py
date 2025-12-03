import io
import re
import tempfile

import pandas as pd
import pdfplumber
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
)
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ============================================
# FastAPI app + CORS
# ============================================

app = FastAPI(title="Wastage Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
#   Helper: money → float
# =========================

def to_num(s):
    """Convert money-like strings to float, safely."""
    if pd.isna(s):
        return 0.0
    s = str(s)
    s = s.replace("£", "").replace(",", "").strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(" ", ""))
        except ValueError:
            return 0.0


# =========================
#  Stock Wastage PDF parser
# =========================

def parse_detail_line(line: str):
    """
    Parse a single product line from the Stock Wastage PDF text using
    right-side extraction.

    Layout (spaces vary):
      PID SUPREF DESCRIPTION ... £cost £retail sales_qty wastage_qty pct £wastage_cost
    """
    u = line.upper().strip()

    # Skip totals / summary / headers (blue rows)
    if "GRAND TOTAL" in u or "SUB DEPARTMENT" in u or u.startswith("TOTAL"):
        return None

    if "£" not in line or "%" not in line:
        return None

    # 1) Extract Wastage Cost (last £amount at end of line)
    m_wc = re.search(r"£\s*([\d,]+\.\d{2})\s*$", line)
    if not m_wc:
        return None
    wastage_cost_str = m_wc.group(1)
    left = line[:m_wc.start()].rstrip()

    # 2) Extract "... sales_qty wastage_qty pct" from the right
    m_q = re.search(r"(\d+)\s+(\d+)\s+(\d+\.\d+%)\s*$", left)
    if not m_q:
        return None
    sales_qty = int(m_q.group(1))
    wastage_qty = int(m_q.group(2))
    pct_str = m_q.group(3)

    left2 = left[:m_q.start()].rstrip()

    # 3) Extract the last two £ amounts from what's left → cost & retail
    money = re.findall(r"£\s*([\d,]+\.\d{2})", left2)
    if len(money) < 2:
        return None

    # In this report the order is: ... £cost £retail ...
    cost_str = money[-2]
    retail_str = money[-1]

    # Remove that price chunk from left2
    all_money_matches = list(re.finditer(r"£\s*([\d,]+\.\d{2})", left2))
    first_of_last_two = all_money_matches[-2]
    left3 = left2[:first_of_last_two.start()].strip()

    tokens = left3.split()
    if len(tokens) < 2:
        return None

    product_id = tokens[0]
    if not re.match(r"^[A-Z]\d{5,}", product_id):
        return None

    supplier_ref = tokens[1] if len(tokens) > 1 else ""
    desc_tokens = tokens[2:]

    # Strip trailing X1 / X2 / X3 etc
    if desc_tokens and re.match(r"^X\d+$", desc_tokens[-1].upper()):
        desc_tokens = desc_tokens[:-1]

    description = " ".join(desc_tokens).strip()
    description = re.sub(r"\s+", " ", description)

    if not description:
        return None

    return {
        "Product ID": product_id,
        "Supplier Ref": supplier_ref,
        "Description": description,
        "Cost Price": cost_str,
        "Retail Price": retail_str,
        "Sales Qty": sales_qty,
        "Wastage Qty": wastage_qty,
        "Wastage %": pct_str,         # original PDF; we recalc later anyway
        "Wastage Cost": wastage_cost_str,
    }


def read_details_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Reads the Stock Wastage PDF as plain text and builds a DataFrame
    by parsing each product line.
    """
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                parsed = parse_detail_line(line)
                if parsed:
                    rows.append(parsed)

    if not rows:
        raise ValueError(f"No product rows parsed from details PDF: {pdf_path}")

    df = pd.DataFrame(rows)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df


# =========================
#  Cashier Override parser
# =========================

def parse_override_line(line: str):
    """
    Parse a line from the Cashier Override PDF.
    Extracts Item Description + Qty.
    """
    u = line.upper().strip()

    # Skip totals / summaries / headers
    if "TOTAL" in u or "SUMMARY" in u:
        return None

    if "£" not in line:
        return None

    tokens = line.split()
    if len(tokens) < 8:
        return None

    # Quantity is usually second last token (e.g. 1.00)
    try:
        qty_str = tokens[-2]
        qty = float(qty_str)
    except ValueError:
        return None

    # Description starts after date,time,employee,txn = index 4
    price_idx = None
    for idx in range(4, len(tokens)):
        if tokens[idx].startswith("£") or tokens[idx].startswith("-£"):
            price_idx = idx
            break

    if price_idx is None or price_idx <= 4:
        return None

    desc_tokens = tokens[4:price_idx]
    description = " ".join(desc_tokens).strip()
    if not description:
        return None

    description = re.sub(r"\s+", " ", description)

    return {"Item Description": description, "Qty": qty}


def read_adjustment_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Reads the Cashier Override PDF and returns DF with:
    - Item Description
    - Qty (float)
    """
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                parsed = parse_override_line(line)
                if parsed:
                    rows.append(parsed)

    if not rows:
        raise ValueError(f"No override rows parsed from adjustment PDF: {pdf_path}")

    df = pd.DataFrame(rows)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df


# =========================
#  Merge logic + meta
# =========================

def merge_from_pdfs(details_pdf_path: str, adjustment_pdf_path: str):
    """
    Returns:
      merged_df, meta_dict

    meta contains (we only really need override_rows for summary tweak):
      - override_rows
    """
    # --- Base wastage data ---
    details_raw = read_details_pdf(details_pdf_path)

    details = details_raw.copy()
    details["Cost Price"] = details["Cost Price"].apply(to_num)
    details["Retail Price"] = details["Retail Price"].apply(to_num)
    details["Sales Qty"] = details["Sales Qty"].astype(float)
    details["Wastage Qty"] = details["Wastage Qty"].astype(float)

    # Base wastage cost and %
    details["Wastage Cost"] = details["Wastage Qty"] * details["Cost Price"]
    denom = details["Sales Qty"] + details["Wastage Qty"]
    details["Wastage %"] = details["Wastage Qty"] / denom.replace(0, pd.NA)
    details["Wastage %"] = details["Wastage %"].fillna(0)

    details["ITEM_KEY"] = details["Description"].astype(str).str.upper().str.strip()

    # --- Overrides ---
    overrides_raw = read_adjustment_pdf(adjustment_pdf_path)
    overrides = overrides_raw.copy()
    overrides["ITEM_KEY"] = overrides["Item Description"].astype(str).str.upper().str.strip()

    override_rows = int(len(overrides))

    # Sum Qty by item (duplicates included in the sum)
    adj_counts = (
        overrides.groupby("ITEM_KEY")["Qty"]
        .sum()
        .reset_index(name="ADJ_QTY")
    )

    # --- Merge base + overrides (only matched ones affect per-item Wastage Qty) ---
    merged = details.merge(adj_counts, on="ITEM_KEY", how="left")
    merged["ADJ_QTY"] = merged["ADJ_QTY"].fillna(0)

    merged["Wastage Qty"] = merged["Wastage Qty"].astype(float) + merged["ADJ_QTY"]

    merged["Wastage Cost"] = merged["Wastage Qty"] * merged["Cost Price"]
    denom2 = merged["Sales Qty"] + merged["Wastage Qty"]
    merged["Wastage %"] = merged["Wastage Qty"] / denom2.replace(0, pd.NA)
    merged["Wastage %"] = merged["Wastage %"].fillna(0)

    meta = {
        "override_rows": override_rows,  # this is what we’ll add on top of total
    }

    return merged, meta


# =========================
#  PDF report generation
# =========================

def generate_wastage_report_from_df(df_num_final: pd.DataFrame, meta: dict) -> bytes:
    styles = getSampleStyleSheet()

    # Prepare display dataframe
    df_disp = df_num_final.copy()

    # Money formatting with £
    df_disp["Cost Price"] = df_num_final["Cost Price"].apply(lambda x: f"£ {x:.2f}")
    df_disp["Wastage Cost"] = df_num_final["Wastage Cost"].apply(lambda x: f"£ {x:.2f}")

    # Wastage % as 0–100%
    df_disp["Wastage %"] = (df_num_final["Wastage %"] * 100).round().astype(int).astype(str) + "%"

    # Quantities as ints
    df_disp["Sales Qty"] = df_num_final["Sales Qty"].round().astype(int)
    df_disp["Wastage Qty"] = df_num_final["Wastage Qty"].round().astype(int)

    # Hide internal columns + Retail Price from tables
    df_disp = df_disp.drop(columns=["ITEM_KEY", "ADJ_QTY", "Retail Price"], errors="ignore")

    # Sort main table by Wastage % desc
    order_idx = df_num_final["Wastage %"].sort_values(ascending=False).index
    sorted_disp = df_disp.loc[order_idx].reset_index(drop=True)
    sorted_raw = df_num_final.loc[order_idx].reset_index(drop=True)

    # Top 20 lists
    top20_wasted_idx = df_num_final["Wastage Qty"].sort_values(ascending=False).index[:20]
    top20_sold_idx = df_num_final["Sales Qty"].sort_values(ascending=False).index[:20]

    top20_wasted_disp = df_disp.loc[top20_wasted_idx]
    top20_wasted_raw = df_num_final.loc[top20_wasted_idx]

    top20_sold_disp = df_disp.loc[top20_sold_idx]
    top20_sold_raw = df_num_final.loc[top20_sold_idx]

    # ======= Summary numbers =======
    total_items = len(df_num_final)

    # Base total wastage from merged per-item table
    base_total_wastage_qty = float(df_num_final["Wastage Qty"].sum())

    # Override rows from Shell Cashier Override (incl. duplicates)
    override_rows = float(meta.get("override_rows", 0))

    # 🔴 This is the change you asked for:
    # Show TOTAL WASTAGE QTY = per-item total + number of override rows
    total_wastage_qty_display = base_total_wastage_qty + override_rows

    total_sales_qty = float(df_num_final["Sales Qty"].sum())

    # Use the adjusted wastage qty for average % (high-level metric)
    if total_sales_qty + total_wastage_qty_display > 0:
        avg_wastage_pct_val = (
            total_wastage_qty_display / (total_sales_qty + total_wastage_qty_display) * 100
        )
    else:
        avg_wastage_pct_val = 0.0

    # Costs/Revenues still based on real per-item data (we don’t guess cost for override-only rows)
    total_wastage_cost_val = float(df_num_final["Wastage Cost"].sum())
    total_revenue_val = float((df_num_final["Retail Price"] * df_num_final["Sales Qty"]).sum())

    # Category counts (still based on per-item percentages)
    wp_pct = df_num_final["Wastage %"] * 100
    high = int((wp_pct >= 70).sum())
    med = int(((wp_pct >= 30) & (wp_pct < 70)).sum())
    low = int(((wp_pct >= 0) & (wp_pct < 30)).sum())

    # Pie chart
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pie_tmp:
        pie_path = pie_tmp.name

    fig, ax = plt.subplots()
    ax.pie(
        [high, med, low],
        labels=["High ≥70%", "Medium 30–70%", "Low 0–30%"],
        autopct="%1.1f%%",
    )
    ax.set_title("Wastage Level Distribution (Item Count)")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close(fig)

    # Build PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=20,
        rightMargin=20,
        topMargin=20,
        bottomMargin=20,
    )
    story = []

    # ---- Summary page ----
    story.append(Paragraph("Wastage Report Summary (From PDFs)", styles["Title"]))
    story.append(Spacer(1, 16))

    # Main summary (with updated TOTAL WASTAGE QTY)
    summary_data = [
        ["TOTAL ITEMS", "TOTAL WASTAGE QTY", "TOTAL SALES QTY",
         "AVG WASTAGE %", "TOTAL WASTAGE COST", "TOTAL REVENUE"],
        [
            str(int(total_items)),
            str(int(round(total_wastage_qty_display))),  # 👈 shows e.g. 705 + 50 = 755
            str(int(round(total_sales_qty))),
            f"{avg_wastage_pct_val:.2f} %",
            f"£ {total_wastage_cost_val:.2f}",
            f"£ {total_revenue_val:.2f}",
        ],
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 14),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Category table
    cat_data = [
        ["CATEGORY", "THRESHOLD", "ITEMS (COUNT)"],
        ["HIGH WASTAGE (≥70%)", "≥ 70%", f"{high} items"],
        ["MEDIUM WASTAGE (30–70%)", "30–70%", f"{med} items"],
        ["LOW WASTAGE (0–30%)", "0–30%", f"{low} items"],
    ]
    cat_table = Table(cat_data)
    cat_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), 11),
        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
    ])
    cat_style.add("BACKGROUND", (0, 1), (-1, 1), colors.Color(1, 0.6, 0.6))
    cat_style.add("BACKGROUND", (0, 2), (-1, 2), colors.Color(1, 1, 0.5))
    cat_style.add("BACKGROUND", (0, 3), (-1, 3), colors.Color(0.7, 1, 0.7))
    cat_table.setStyle(cat_style)
    story.append(cat_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Wastage Distribution by Category (Item Count)", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Image(pie_path, width=400, height=250))
    story.append(PageBreak())

    # ---- Helper for big tables ----
    def add_full_table(title, df_disp_sub, df_raw_sub, rows_per_page=50):
        total = len(df_disp_sub)
        for start in range(0, total, rows_per_page):
            end = min(start + rows_per_page, total)
            chunk_disp = df_disp_sub.iloc[start:end]
            chunk_raw = df_raw_sub.iloc[start:end]

            story.append(Paragraph(f"{title} (Rows {start + 1}–{end})", styles["Heading2"]))
            story.append(Spacer(1, 10))

            table_data = [list(chunk_disp.columns)] + chunk_disp.values.tolist()
            tbl = Table(table_data, repeatRows=1)

            style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])

            for i, wp in enumerate((chunk_raw["Wastage %"] * 100).tolist(), start=1):
                if wp >= 70:
                    bg = colors.Color(1, 0.6, 0.6)
                elif wp >= 30:
                    bg = colors.Color(1, 1, 0.5)
                else:
                    bg = colors.Color(0.7, 1, 0.7)
                style.add("BACKGROUND", (0, i), (-1, i), bg)

            tbl.setStyle(style)
            story.append(tbl)
            story.append(PageBreak())

    add_full_table("All Items Sorted by Wastage % (High → Low)", sorted_disp, sorted_raw)

    # ---- Top 20 tables ----
    def add_top20_table(title, disp_sub, raw_sub):
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 10))

        table_data = [list(disp_sub.columns)] + disp_sub.values.tolist()
        tbl = Table(table_data, repeatRows=1)

        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])

        for i, wp in enumerate((raw_sub["Wastage %"] * 100).tolist(), start=1):
            if wp >= 70:
                bg = colors.Color(1, 0.6, 0.6)
            elif wp >= 30:
                bg = colors.Color(1, 1, 0.5)
            else:
                bg = colors.Color(0.7, 1, 0.7)
            style.add("BACKGROUND", (0, i), (-1, i), bg)

        tbl.setStyle(style)
        story.append(tbl)
        story.append(PageBreak())

    add_top20_table("Top 20 Most Wasted Items (by Wastage Qty)", top20_wasted_disp, top20_wasted_raw)
    add_top20_table("Top 20 Most Sold Items (by Sales Qty)", top20_sold_disp, top20_sold_raw)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# =========================
#  FastAPI endpoint
# =========================

@app.post("/generate-report")
async def generate_report(
    details_pdf: UploadFile = File(...),
    adjustment_pdf: UploadFile = File(...),
):
    if details_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="details_pdf must be a PDF")
    if adjustment_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="adjustment_pdf must be a PDF")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_details:
        details_path = tmp_details.name
        tmp_details.write(await details_pdf.read())

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_adjust:
        adjust_path = tmp_adjust.name
        tmp_adjust.write(await adjustment_pdf.read())

    try:
        merged_df, meta = merge_from_pdfs(details_path, adjust_path)
        pdf_bytes = generate_wastage_report_from_df(merged_df, meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Wastage_analysis_from_pdfs.pdf"},
    )