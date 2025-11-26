import re
import io
import tempfile

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

import pdfplumber

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, PageBreak, Image
)
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


app = FastAPI(title="Wastage Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
#  Helpers: PDF → DataFrame
# =========================

def extract_table_with_column(pdf_path: str, col_keyword: str) -> pd.DataFrame:
    """
    Open a PDF, scan all tables on all pages, and return a concatenated DataFrame
    only for tables that contain a header with the given keyword.
    """
    frames = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = [(h or "").strip() for h in table[0]]
                # normalise header whitespace
                header_norm = [re.sub(r"\s+", " ", h) for h in header]

                if any(col_keyword.lower() in h.lower() for h in header_norm):
                    df = pd.DataFrame(table[1:], columns=header_norm)
                    frames.append(df)

    if not frames:
        raise ValueError(f"No tables with column containing '{col_keyword}' in {pdf_path}")

    return pd.concat(frames, ignore_index=True)


def parse_detail_line(line: str):
    """
    Parse a single product line from the Stock Wastage PDF text.
    Example:
    C0000003275 607600 WR BRIE BACON & CHILLI RELISH X1 £2.98 £4.55 11 2 15.38% £5.96
    """

    u = line.upper().strip()

    # 🔵 Skip "TOTAL" / summary lines (blue ones in PDF)
    if "TOTAL" in u or "SUBTOTAL" in u:
        return None

    # We expect product rows to start with a product code like C000000...
    tokens = line.split()
    if len(tokens) < 8:
        return None

    first = tokens[0]
    # Require a product-id-like token (starts with a letter + digits)
    if not re.match(r"^[A-Z]\d{5,}", first):
        return None

    if "£" not in line or "%" not in line:
        return None

    try:
        wastage_cost_str = tokens[-1]
        pct_str          = tokens[-2]
        wastage_qty      = int(tokens[-3])
        sales_qty        = int(tokens[-4])
        retail_str       = tokens[-5]
        cost_str         = tokens[-6]
    except ValueError:
        return None

    product_id = tokens[0]
    supplier_ref = tokens[1] if len(tokens) > 7 else ""

    desc_tokens = tokens[2:-6]
    if desc_tokens and desc_tokens[-1] == "X1":
        desc_tokens = desc_tokens[:-1]
    description = " ".join(desc_tokens).strip()

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
        "Wastage %": pct_str,
        "Wastage Cost": wastage_cost_str,
    }

def parse_override_line(line: str):
    """
    Parse a line from the Cashier Override PDF.
    We only care about the 'Item Description' text.
    """

    u = line.upper().strip()

    # 🔵 Skip "TOTAL" / summary / footer lines
    if "TOTAL" in u or "SUMMARY" in u:
        return None

    if "£" not in line:
        return None

    tokens = line.split()
    if len(tokens) < 8:
        return None

    # Description starts at index 4 (after date, time, employee, txn)
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
    return {"Item Description": description}


def read_details_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Reads the Stock Wastage PDF as plain text and builds a DataFrame
    by parsing each product line (no table detection needed).
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
    # Normalise column names (just in case)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df


def read_adjustment_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Reads the Cashier Override PDF as plain text and builds a DataFrame
    containing a single 'Item Description' column.
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
        # not fatal logically, but keep consistent error style
        raise ValueError(f"No override rows parsed from adjustment PDF: {pdf_path}")

    df = pd.DataFrame(rows)
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df

def find_col(cols, keyword: str):
    for c in cols:
        if keyword.lower() in str(c).lower():
            return c
    return None


def to_num(s):
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
#  Merge logic using Description
# =========================

def merge_from_pdfs(details_pdf_path: str, adjustment_pdf_path: str) -> pd.DataFrame:
    # --- Read & clean details (stock wastage) ---
    details_raw = read_details_pdf(details_pdf_path)
    details_cols = list(details_raw.columns)

    desc_col   = find_col(details_cols, "Description")
    retail_col = find_col(details_cols, "Retail")
    cost_col   = find_col(details_cols, "Cost")
    sales_col  = find_col(details_cols, "Sales")
    wq_col     = find_col(details_cols, "Wastage Qty")

    if not all([desc_col, retail_col, cost_col, sales_col, wq_col]):
        raise RuntimeError(
            f"Could not detect all required columns from details PDF. Got: {details_cols}"
        )

    df_num = pd.DataFrame()
    df_num["Description"]   = details_raw[desc_col].astype(str).str.strip()
    df_num["Retail Price"]  = details_raw[retail_col].apply(to_num)
    df_num["Cost Price"]    = details_raw[cost_col].apply(to_num)
    df_num["Sales Qty"]     = details_raw[sales_col].apply(
        lambda x: float(str(x).strip() or 0)
    )
    df_num["Wastage Qty"]   = details_raw[wq_col].apply(
        lambda x: float(str(x).strip() or 0)
    )

    # Normalised key used for merge
    df_num["ITEM_KEY"] = df_num["Description"].astype(str).str.upper().str.strip()

    # --- Read & clean adjustment (cashier overrides) ---
    adjust_raw = read_adjustment_pdf(adjustment_pdf_path)
    adj_cols = list(adjust_raw.columns)
    item_col = find_col(adj_cols, "Item Description")

    if item_col is None:
        raise RuntimeError(
            f"Could not find 'Item Description' column in adjustment PDF. Got: {adj_cols}"
        )

    adjust_df = pd.DataFrame()
    adjust_df["ITEM_KEY"] = adjust_raw[item_col].astype(str).str.upper().str.strip()

    # Each row in adjustment = +1 wastage unit for that item
    adj_counts = adjust_df.groupby("ITEM_KEY").size().reset_index(name="ADJ_COUNT")

    # Merge counts into main df
    merged = df_num.merge(adj_counts, on="ITEM_KEY", how="left")
    merged["ADJ_COUNT"] = merged["ADJ_COUNT"].fillna(0)

    # Update Wastage Qty
    merged["Wastage Qty"] = merged["Wastage Qty"].astype(float) + merged["ADJ_COUNT"]

    # Recalculate Wastage Cost & Wastage %
    merged["Wastage Cost"] = merged["Wastage Qty"] * merged["Cost Price"]
    denom = merged["Sales Qty"] + merged["Wastage Qty"]
    merged["Wastage %"] = merged["Wastage Qty"] / denom.replace(0, pd.NA)
    merged["Wastage %"] = merged["Wastage %"].fillna(0)

    return merged


# =========================
#  Report generation
# =========================

def generate_wastage_report_from_df(df_num_final: pd.DataFrame) -> bytes:
    styles = getSampleStyleSheet()

    # Display frame with formatting
    df_disp = df_num_final.copy()
    df_disp["Cost Price"]   = df_num_final["Cost Price"].apply(lambda x: f"£ {x:.2f}")
    df_disp["Wastage Cost"] = df_num_final["Wastage Cost"].apply(lambda x: f"£ {x:.2f}")
    df_disp["Wastage %"]    = (
        df_num_final["Wastage %"] * 100
    ).round().astype(int).astype(str) + "%"
        # Format quantities as whole numbers
    df_disp["Sales Qty"]   = df_num_final["Sales Qty"].round().astype(int)
    df_disp["Wastage Qty"] = df_num_final["Wastage Qty"].round().astype(int)

    # Drop internal / unwanted columns from all tables
    df_disp = df_disp.drop(
        columns=["ITEM_KEY", "ADJ_COUNT", "Retail Price"],
        errors="ignore",
    )

    # Sort by Wastage % DESC
    order_idx   = df_num_final["Wastage %"].sort_values(ascending=False).index
    sorted_disp = df_disp.loc[order_idx].reset_index(drop=True)
    sorted_raw  = df_num_final.loc[order_idx].reset_index(drop=True)

    # Top 20 most wasted / most sold
    top20_wasted_idx = df_num_final["Wastage Qty"].sort_values(ascending=False).index[:20]
    top20_sold_idx   = df_num_final["Sales Qty"].sort_values(ascending=False).index[:20]

    top20_wasted_disp = df_disp.loc[top20_wasted_idx]
    top20_wasted_raw  = df_num_final.loc[top20_wasted_idx]

    top20_sold_disp = df_disp.loc[top20_sold_idx]
    top20_sold_raw  = df_num_final.loc[top20_sold_idx]

    # Summary numbers
    total_items       = len(df_num_final)
    total_wastage_qty = df_num_final["Wastage Qty"].sum()
    total_sales_qty   = df_num_final["Sales Qty"].sum()

    if total_sales_qty + total_wastage_qty > 0:
        avg_wastage_pct_val = total_wastage_qty / (total_sales_qty + total_wastage_qty) * 100
    else:
        avg_wastage_pct_val = 0.0

    total_wastage_cost_val = df_num_final["Wastage Cost"].sum()
    total_revenue_val      = (df_num_final["Retail Price"] * df_num_final["Sales Qty"]).sum()

    # Category counts
    wp_pct = df_num_final["Wastage %"] * 100
    high = int((wp_pct >= 70).sum())
    med  = int(((wp_pct >= 30) & (wp_pct < 70)).sum())
    low  = int(((wp_pct >= 0) & (wp_pct < 30)).sum())

    # Pie chart → temp PNG
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

    # Generate PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=20, rightMargin=20,
        topMargin=20, bottomMargin=20,
    )
    story = []

    # ---- Summary Page ----
    story.append(Paragraph("Wastage Report Summary (From PDFs)", styles["Title"]))
    story.append(Spacer(1, 16))

    summary_data = [
        ["TOTAL ITEMS", "TOTAL WASTAGE QTY", "TOTAL SALES QTY",
         "AVG WASTAGE %", "TOTAL WASTAGE COST", "TOTAL REVENUE"],
        [
            str(int(total_items)),
            str(int(total_wastage_qty)),
            str(int(total_sales_qty)),
            f"{avg_wastage_pct_val:.2f} %",
            f"£ {total_wastage_cost_val:.2f}",
            f"£ {total_revenue_val:.2f}",
        ],
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgrey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 14),
        ("FONTNAME",   (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 1), (-1, 1), 13),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.6, colors.black),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    cat_data = [
        ["CATEGORY", "THRESHOLD", "ITEMS (COUNT)"],
        ["HIGH WASTAGE (≥70%)", "≥ 70%", f"{high} items"],
        ["MEDIUM WASTAGE (30–70%)", "30–70%", f"{med} items"],
        ["LOW WASTAGE (0–30%)", "0–30%", f"{low} items"],
    ]
    cat_table = Table(cat_data)
    cat_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgrey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 12),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 1), (-1, -1), 11),
        ("ALIGN",      (0, 1), (-1, -1), "CENTER"),
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

    # ---- Full table (50 rows per page) ----
    def add_full_table(title, df_disp_sub, df_raw_sub, rows_per_page=50):
        total = len(df_disp_sub)
        for start in range(0, total, rows_per_page):
            end = min(start + rows_per_page, total)
            chunk_disp = df_disp_sub.iloc[start:end]
            chunk_raw  = df_raw_sub.iloc[start:end]

            story.append(Paragraph(f"{title} (Rows {start+1}–{end})", styles["Heading2"]))
            story.append(Spacer(1, 10))

            table_data = [list(chunk_disp.columns)] + chunk_disp.values.tolist()
            tbl = Table(table_data, repeatRows=1)

            style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, 0), 11),
                ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",   (0, 1), (-1, -1), 9),
                ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
                ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
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
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 11),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 1), (-1, -1), 9),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
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

    # Build into buffer
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

    # Save uploaded PDFs to temp files
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_details:
        details_path = tmp_details.name
        tmp_details.write(await details_pdf.read())

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_adjust:
        adjust_path = tmp_adjust.name
        tmp_adjust.write(await adjustment_pdf.read())

    try:
        merged_df = merge_from_pdfs(details_path, adjust_path)
        pdf_bytes = generate_wastage_report_from_df(merged_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Wastage_analysis_from_pdfs.pdf"},
    )