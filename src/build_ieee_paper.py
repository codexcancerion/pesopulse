"""
Builds the 6-Page IEEE Technical Project Paper (.docx and .md) for:
PesoPulse (MIT 118 Final Group Project, Team 3):
Forecasting Philippine Provincial Inflation and Supply-Shock Pass-Through.

Editorial and Formatting Guarantees:
- Classic IEEE two-column layout (US Letter, Times New Roman, Roman numeral sections)
- Plain, clear storytelling with minimal jargon, zero em dashes, and minimal parentheses
- Short, straightforward discussions aligned strictly with Rubric Section 7.3
- 100% verified metrics from PesoPulse data artifacts and live app.py simulation
- Zero raw Markdown, HTML tags, XML entities, or LaTeX symbols in the Word (.docx) output
"""

from __future__ import annotations

import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
FIG_DIR = BASE_DIR / "figures"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

NAVY = RGBColor(0x0F, 0x2B, 0x4C)
SLATE = RGBColor(0x33, 0x41, 0x55)
DARK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x47, 0x55, 0x69)


# ==============================================================================
# 1. GENERATE CLEAN, SIMPLE PIPELINE ARCHITECTURE DIAGRAM
# ==============================================================================

def build_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 2.55), dpi=300)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 2.55)
    ax.axis("off")

    stages = [
        ("1. Raw Government Data", "PSA OpenSTAT\n8 Official Tables\n4.5M Raw CPI Cells\n73.87 MB Source", "#EFF6FF", "#1D4ED8"),
        ("2. DuckDB SQL ETL", "Unpivot 106 Months\nStandardize 101 Geos\nJoin Weights & Rice\nWindow Lag Features", "#F0FDFA", "#0D9488"),
        ("3. Compact Parquet", "ZSTD Columnar Marts\nPanel: 10,706 Rows\nSize: 2.2 MB Total\nFast Cloud Loading", "#FEFCE8", "#CA8A04"),
        ("4. Forecasting Engine", "Ridge & XGBoost ML\nQuantile 80/95% CI\nSARIMAX Fan Chart\nTreeSHAP Explain", "#FFF7ED", "#EA580C"),
        ("5. Live Streamlit App", "DuckDB Queries Only\n<1,000 Rows per Chart\n12 Interactive Views\nWhat-If Shock Engine", "#FEF2F2", "#DC2626"),
    ]

    box_w = 1.92
    box_h = 1.82
    y0 = 0.36
    xs = [0.15, 2.38, 4.61, 6.84, 9.07]

    for idx, (x, (title, body, bg, edge)) in enumerate(zip(xs, stages)):
        rect = patches.FancyBboxPatch(
            (x, y0), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=bg, edgecolor=edge, linewidth=1.7,
        )
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch(
            (x, y0 + box_h - 0.42), box_w, 0.42,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=edge, edgecolor=edge, linewidth=0.5,
        )
        ax.add_patch(hdr)
        ax.text(
            x + box_w / 2, y0 + box_h - 0.21, title,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="white",
            family="sans-serif",
        )
        ax.text(
            x + box_w / 2, y0 + 0.68, body,
            ha="center", va="center", fontsize=8.0, color="#1E293B", linespacing=1.45,
            family="sans-serif",
        )
        if idx < len(xs) - 1:
            ax.annotate(
                "", xy=(xs[idx + 1] - 0.08, y0 + box_h / 2),
                xytext=(x + box_w + 0.08, y0 + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.8, mutation_scale=13),
            )

    ax.text(
        5.6, 2.40,
        "PesoPulse End-to-End Inflation Analytics Pipeline: PSA OpenSTAT to Live Decision Support",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0F172A", family="sans-serif",
    )
    plt.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==============================================================================
# 2. IEEE TWO-COLUMN WORD BUILDER
# ==============================================================================

def set_section_columns(section, num_cols: int, space_in: float = 0.24):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), str(int(space_in * 1440)))


def set_cell_shading(cell, fill_hex: str):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=38, bottom=38, left=55, right=55):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def set_cell_width_dxa(cell, width_in: float):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcW")):
        tcPr.remove(old)
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(int(round(width_in * 1440))))
    tcW.set(qn("w:type"), "dxa")
    tcPr.append(tcW)


def set_table_fixed_layout(table, col_widths: list[float]):
    total_dxa = int(round(sum(col_widths) * 1440))
    tblPr = table._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        for old in tblPr.findall(qn(tag)):
            tblPr.remove(old)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(total_dxa))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    tblLayout = OxmlElement("w:tblLayout")
    tblLayout.set(qn("w:type"), "fixed")
    tblPr.append(tblLayout)

    tblGrid = table._tbl.tblGrid
    if tblGrid is not None:
        for old in list(tblGrid):
            tblGrid.remove(old)
        for w in col_widths:
            gc = OxmlElement("w:gridCol")
            gc.set(qn("w:w"), str(int(round(w * 1440))))
            tblGrid.append(gc)


def set_ieee_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="8" w:space="0" w:color="0F2B4C"/>'
        '  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="0F2B4C"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '  <w:left w:val="none"/>'
        '  <w:right w:val="none"/>'
        '  <w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)


def remove_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="none"/><w:bottom w:val="none"/>'
        '  <w:left w:val="none"/><w:right w:val="none"/>'
        '  <w:insideH w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)


def add_page_number_field(paragraph):
    run = paragraph.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(8.0)
    run.font.color.rgb = MUTED
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def add_inline_runs(paragraph, text: str, font_size: float = 9.5, color=DARK):
    pattern = re.compile(r"(<b><i>.*?</i></b>|<b>.*?</b>|<i>.*?</i>)")
    for part in pattern.split(text):
        if not part:
            continue
        bold, italic, clean = False, False, part
        if part.startswith("<b><i>") and part.endswith("</i></b>"):
            bold, italic, clean = True, True, part[6:-8]
        elif part.startswith("<b>") and part.endswith("</b>"):
            bold, clean = True, part[3:-4]
        elif part.startswith("<i>") and part.endswith("</i>"):
            italic, clean = True, part[3:-4]
        run = paragraph.add_run(clean)
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        run.bold = bold
        run.italic = italic


def html_to_md(text: str) -> str:
    s = re.sub(r"<b><i>(.*?)</i></b>", r"***\1***", text)
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s)
    s = re.sub(r"<i>(.*?)</i>", r"*\1*", s)
    return s


class IEEEPaperBuilder:
    def __init__(self):
        self.doc = Document()
        self.md_lines: list[str] = []
        self.body_texts: list[str] = []
        self._init_page()

    def _init_page(self):
        sec = self.doc.sections[0]
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)
        sec.top_margin = Inches(0.60)
        sec.bottom_margin = Inches(0.60)
        sec.left_margin = Inches(0.62)
        sec.right_margin = Inches(0.62)
        set_section_columns(sec, 1)

        normal = self.doc.styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(9.5)
        normal.font.color.rgb = DARK

        footer = sec.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr1 = fp.add_run("PesoPulse  |  MIT 118 Technical Project Paper (Team 3)  |  Page ")
        fr1.font.name = "Times New Roman"
        fr1.font.size = Pt(8.0)
        fr1.font.color.rgb = MUTED
        add_page_number_field(fp)

    def switch_columns(self, num_cols: int):
        sec = self.doc.add_section(WD_SECTION_START.CONTINUOUS)
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)
        sec.top_margin = Inches(0.60)
        sec.bottom_margin = Inches(0.60)
        sec.left_margin = Inches(0.62)
        sec.right_margin = Inches(0.62)
        sec.footer.is_linked_to_previous = True
        set_section_columns(sec, num_cols, space_in=0.24)

    def add_ieee_header(self, title: str, authors: list[tuple[str, str, str]]):
        p_t = self.doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.paragraph_format.space_before = Pt(0)
        p_t.paragraph_format.space_after = Pt(5)
        r_t = p_t.add_run(title)
        r_t.font.name = "Times New Roman"
        r_t.font.size = Pt(17)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY

        tbl = self.doc.add_table(rows=1, cols=len(authors))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        remove_table_borders(tbl)
        for i, (name, dept, loc) in enumerate(authors):
            cell = tbl.rows[0].cells[i]
            set_cell_margins(cell, top=16, bottom=24, left=35, right=35)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(1)
            r1 = p.add_run(name + "\n")
            r1.font.name = "Times New Roman"
            r1.font.size = Pt(10)
            r1.font.bold = True
            r1.font.color.rgb = DARK

            r2 = p.add_run(dept + "\n" + loc)
            r2.font.name = "Times New Roman"
            r2.font.size = Pt(8.5)
            r2.font.italic = True
            r2.font.color.rgb = SLATE

        p_sp = self.doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(2)

        self.md_lines.append(f"# {title}\n")
        author_str = ", ".join(a[0] for a in authors)
        self.md_lines.append(f"**Authors:** {author_str}  ")
        self.md_lines.append("**Affiliation:** School of Advanced Studies, Saint Louis University, Baguio City, Philippines  ")
        self.md_lines.append("**Course:** MIT 118 Data Visualization and Visual Analytics, Team 3\n")
        self.md_lines.append("---\n")

    def add_abstract(self, abstract_text: str, keywords: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.07
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3.5)
        add_inline_runs(p, f"<b><i>Abstract:</i></b> {abstract_text}", font_size=9.0, color=DARK)

        pk = self.doc.add_paragraph()
        pk.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pk.paragraph_format.line_spacing = 1.07
        pk.paragraph_format.space_after = Pt(5)
        add_inline_runs(pk, f"<b><i>Index Terms:</i></b> {keywords}", font_size=8.8, color=SLATE)

        self.body_texts.append(abstract_text)
        self.md_lines.append(f"## Abstract\n\n{html_to_md(abstract_text)}\n")
        self.md_lines.append(f"**Index Terms:** {keywords}\n")

    def add_section_heading(self, text: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(3.5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text.upper())
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.0)
        r.font.bold = True
        r.font.color.rgb = NAVY
        self.md_lines.append(f"\n## {text}\n")

    def add_sub_heading(self, text: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(5.5)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = SLATE
        self.md_lines.append(f"\n### {text}\n")

    def add_para(self, text: str, space_after: float = 4.2):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.09
        p.paragraph_format.space_after = Pt(space_after)
        add_inline_runs(p, text, font_size=9.5, color=DARK)
        self.body_texts.append(text)
        self.md_lines.append(f"{html_to_md(text)}\n")

    def add_bullets(self, items: list[str]):
        for item in items:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.16)
            p.paragraph_format.first_line_indent = Inches(-0.10)
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.space_after = Pt(2.8)
            r_b = p.add_run("• ")
            r_b.font.name = "Times New Roman"
            r_b.font.size = Pt(9.3)
            r_b.font.bold = True
            r_b.font.color.rgb = NAVY
            add_inline_runs(p, item, font_size=9.3, color=DARK)
            self.body_texts.append(item)
            self.md_lines.append(f"- {html_to_md(item)}")
        self.md_lines.append("")

    def add_table(
        self,
        label: str,
        title: str,
        headers: list[str],
        rows: list,
        col_widths: list[float] | None = None,
        right_cols: set[int] | None = None,
        font_size: float = 7.9,
        keep_together: bool = True,
    ):
        """
        Supports normal list[str] rows OR ("BANNER", "Panel Title") merged banner rows.
        Sets keep_with_next across rows so compact tables never awkwardly split across columns.
        """
        right_cols = right_cols or set()
        p_l = self.doc.add_paragraph()
        p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_l.paragraph_format.space_before = Pt(4.5)
        p_l.paragraph_format.space_after = Pt(0.5)
        p_l.paragraph_format.keep_with_next = True
        rl = p_l.add_run(label.upper())
        rl.font.name = "Times New Roman"
        rl.font.size = Pt(8.5)
        rl.font.bold = True
        rl.font.color.rgb = NAVY

        p_t = self.doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.paragraph_format.space_before = Pt(0)
        p_t.paragraph_format.space_after = Pt(2.5)
        p_t.paragraph_format.keep_with_next = True
        rt = p_t.add_run(title)
        rt.font.name = "Times New Roman"
        rt.font.size = Pt(8.4)
        rt.font.italic = True
        rt.font.color.rgb = SLATE

        table = self.doc.add_table(rows=len(rows) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        set_ieee_table_borders(table)
        if col_widths:
            set_table_fixed_layout(table, col_widths)

        # Header row
        set_row_cant_split(table.rows[0])
        for j, h in enumerate(headers):
            c = table.rows[0].cells[j]
            if col_widths:
                set_cell_width_dxa(c, col_widths[j])
            set_cell_shading(c, "0F2B4C")
            set_cell_margins(c, top=38, bottom=38, left=40, right=40)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            if keep_together:
                p.paragraph_format.keep_with_next = True
            r = p.add_run(h)
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        # Data rows
        for i, row in enumerate(rows):
            t_row = table.rows[i + 1]
            set_row_cant_split(t_row)
            is_last = (i == len(rows) - 1)

            if isinstance(row, tuple) and row[0] == "BANNER":
                merged = t_row.cells[0]
                for j in range(1, len(headers)):
                    merged = merged.merge(t_row.cells[j])
                if col_widths:
                    set_cell_width_dxa(merged, sum(col_widths))
                set_cell_shading(merged, "E2E8F0")
                set_cell_margins(merged, top=34, bottom=34, left=40, right=40)
                p = merged.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                if keep_together and not is_last:
                    p.paragraph_format.keep_with_next = True
                add_inline_runs(p, f"<b>{row[1]}</b>", font_size=font_size, color=NAVY)
                continue

            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            for j, val in enumerate(row):
                c = t_row.cells[j]
                if col_widths:
                    set_cell_width_dxa(c, col_widths[j])
                set_cell_shading(c, bg)
                set_cell_margins(c, top=32, bottom=32, left=40, right=40)
                c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_cols else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                if keep_together and not is_last:
                    p.paragraph_format.keep_with_next = True
                add_inline_runs(p, val, font_size=font_size, color=DARK)

        p_sp = self.doc.add_paragraph()
        p_sp.paragraph_format.space_after = Pt(3.5)

        # Markdown mirror
        self.md_lines.append(f"\n**{label}. {title}**\n")
        self.md_lines.append("| " + " | ".join(headers) + " |")
        self.md_lines.append("| " + " | ".join(["---:" if j in right_cols else ":---" for j in range(len(headers))]) + " |")
        for row in rows:
            if isinstance(row, tuple) and row[0] == "BANNER":
                self.md_lines.append(f"| **{row[1]}** |" + " |" * (len(headers) - 1))
            else:
                self.md_lines.append("| " + " | ".join(html_to_md(v) for v in row) + " |")
        self.md_lines.append("")

    def add_fullwidth_figure(self, img_path: Path, caption: str, width_in: float = 6.50):
        self.switch_columns(1)
        p_img = self.doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(2)
        p_img.paragraph_format.space_after = Pt(1.5)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

        p_cap = self.doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0.5)
        p_cap.paragraph_format.space_after = Pt(3.5)
        add_inline_runs(p_cap, caption, font_size=8.3, color=SLATE)
        self.switch_columns(2)

        rel = f"../figures/{img_path.name}"
        self.md_lines.append(f"\n![{html_to_md(caption)}]({rel})\n")
        self.md_lines.append(f"*{html_to_md(caption)}*\n")

    def add_references(self, refs: list[str]):
        self.add_section_heading("References")
        for idx, ref in enumerate(refs, start=1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.22)
            p.paragraph_format.first_line_indent = Inches(-0.22)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_after = Pt(2.4)
            add_inline_runs(p, f"[{idx}] {ref}", font_size=8.3, color=DARK)
            self.md_lines.append(f"[{idx}] {html_to_md(ref)}")
        self.md_lines.append("")

    def save_and_audit(self, docx_path: Path, md_path: Path):
        self.doc.save(str(docx_path))
        md_path.write_text("\n".join(self.md_lines), encoding="utf-8")

        # Audit actual rendered Word text across paragraphs and tables
        all_rendered = [p.text for p in self.doc.paragraphs]
        for t in self.doc.tables:
            for r in t.rows:
                for c in r.cells:
                    all_rendered.append(c.text)
        joined_doc = " ".join(all_rendered)

        full_body = " ".join(self.body_texts)
        em_dashes = full_body.count("—")
        en_dashes = full_body.count("–")
        parens = full_body.count("(")
        bad_tags = len(re.findall(r"</?[bi]>|&gt;|&lt;|\*\*|`|\$", joined_doc))
        word_count = len(re.sub(r"<.*?>", "", full_body).split())
        print(f"Saved: {docx_path.name} and {md_path.name}")
        print(
            f"Style Audit -> Body Words: {word_count} | Em dashes: {em_dashes} | "
            f"En dashes: {en_dashes} | Parentheses '(': {parens} | Bad tags/entities in DOCX: {bad_tags}"
        )


# ==============================================================================
# 3. COMPOSE THE 6-PAGE IEEE TECHNICAL PAPER (SIMPLE, BRIEF, CLEAR STORYTELLING)
# ==============================================================================

def main():
    pipe_fig = FIG_DIR / "fig00_pipeline_architecture.png"
    build_pipeline_diagram(pipe_fig)

    b = IEEEPaperBuilder()
    b.add_ieee_header(
        title="PesoPulse: Forecasting Philippine Provincial Inflation and Supply-Shock Pass-Through Using DuckDB and Visual Analytics",
        authors=[
            ("Melbert Marafo", "School of Advanced Studies", "Saint Louis University, Baguio City"),
            ("Jhaynher Felix", "School of Advanced Studies", "Saint Louis University, Baguio City"),
            ("David Justine Javillonar", "School of Advanced Studies", "Saint Louis University, Baguio City"),
        ],
    )

    b.switch_columns(2)

    b.add_abstract(
        "A single national inflation number hides very different price realities across Philippine provinces. Families in island and rural areas often face much steeper food costs than shoppers in Metro Manila, and low-income households absorb the hardest hit whenever rice prices jump. We built PesoPulse, an end-to-end visual analytics and forecasting platform using eight official Philippine Statistics Authority datasets covering 101 provinces and highly urbanized cities from January 2018 to October 2026. Our DuckDB pipeline reshapes 4.5 million raw price cells into compact Parquet tables totaling 2.2 MB, feeding a live Streamlit dashboard with twelve interactive charts and a supply-shock simulator. Regional inflation differs significantly at p < 0.001, with Region V averaging 4.39% against 3.41% in NCR. During food spikes above 5%, bottom-30% households face 0.95 percentage points higher inflation than the general population. On 2025 to 2026 holdout data, Ridge Regression delivers the best next-month forecast with 0.93 percentage points MAE and R² of 0.874, while XGBoost flags breaches above the 4.0% central bank ceiling at 92.6% accuracy and 0.976 ROC-AUC.",
        "Consumer Price Index, DuckDB, Inflation Forecasting, PesoPulse, Philippine Statistics Authority, Streamlit, Visual Analytics.",
    )

    # ------------------------------------------------------------------
    # SECTION I
    # ------------------------------------------------------------------
    b.add_section_heading("I. Executive Summary and Public Sector Problem Statement")

    b.add_para(
        "Every month, headlines report one national inflation rate for the Philippines. That single average looks simple, yet it misses how unevenly prices rise from one province to another. Between January 2019 and October 2026, national inflation surged to 8.7% in January 2023, cooled to 0.9% in July 2025, and climbed back to 7.8% in October 2026. By October 2026, one peso from 2018 bought only 0.72 pesos of goods on average across provinces, dropping as low as 0.66 pesos in MIMAROPA, Cagayan Valley, and Caraga compared with 0.75 pesos in NCR."
    )

    b.add_para(
        "Three public sector agencies face real operational gaps when monitoring these price shifts. First, regional offices of the Bangko Sentral ng Pilipinas need early warning before local inflation crosses the 4.0% upper policy target [7]. Second, Regional Development Councils under NEDA and DEPDev need to see which provinces suffer persistent cost pressure so budget and logistics support go to the right areas. Third, local price councils under DTI and the Department of Agriculture need to estimate how a sudden jump in rice, electricity, or diesel prices will hit ordinary and low-income households next month."
    )

    b.add_para(
        "Public CPI files from the Philippine Statistics Authority are large, wide spreadsheets spread across separate portals for headline prices, basket weights, bottom-30% households, wholesale prices, and retail rice quotes. Local planners cannot easily combine these files, run statistical checks, or test price shock scenarios without specialized tools. PesoPulse bridges that gap by turning eight official government tables into one fast, transparent web platform."
    )

    b.add_fullwidth_figure(
        FIG_DIR / "fig01_national_headline_vs_core_vs_b30.png",
        "<b>Fig. 1.</b> Philippine monthly headline, core, and bottom-30% inflation from January 2019 to October 2026 against the 2.0% to 4.0% BSP target band.",
        width_in=6.55,
    )

    # ------------------------------------------------------------------
    # SECTION II
    # ------------------------------------------------------------------
    b.add_section_heading("II. Pipeline Architecture")

    b.add_sub_heading("A. Government Data Sources and Schema Design")
    b.add_para(
        "All inputs come directly from verified Philippine Statistics Authority OpenSTAT tables [1]. The core file is the monthly Consumer Price Index release with 2018 as base year, covering 120 geographic entries and 354 spending items across 106 valid months from January 2018 to October 2026. That single wide file holds 42,480 rows and 4,502,880 monthly price cells at 73.87 MB. To give the forecasting models real economic context, we paired it with seven companion OpenSTAT tables summarized in Table I."
    )

    b.add_table(
        label="TABLE I",
        title="Verified PSA OpenSTAT Data Sources Joined in DuckDB",
        headers=["Source Dataset", "Raw Scale", "Analytical Role"],
        rows=[
            ["Monthly CPI 2018=100", "4.50M cells", "Base index across 120 geos"],
            ["Official CPI YoY %", "230,400 rows", "Verified inflation targets"],
            ["2018 Basket Weights", "42,480 rows", "Local category share %"],
            ["Purchasing Power PPP", "12,960 rows", "Real value of 2018 peso"],
            ["National Core CPI", "1,632 rows", "Non-food price baseline"],
            ["Bottom-30% CPI & Wts", "104,760 rows", "Low-income inflation gap"],
            ["Wholesale GWPI", "1,296 rows", "Upstream island cost push"],
            ["Retail Rice Prices", "25,704 rows", "Local rice PHP per kg"],
        ],
        col_widths=[1.26, 0.82, 1.38],
        font_size=7.8,
    )

    b.add_sub_heading("B. DuckDB SQL Transformation and Data Hygiene")
    b.add_para(
        "Loading millions of raw CSV cells straight into Pandas slows down dashboards and wastes memory. Instead, our ETL script runs all heavy work inside DuckDB, an in-memory columnar SQL engine [2]. DuckDB unpivots the 106 monthly columns, converts PSA placeholder dots into proper nulls, and strips national and regional summary rows so only 101 clean local leaf units remain: 100 provinces and highly urbanized cities plus NCR across 18 regions, including Negros Island Region and BARMM."
    )

    b.add_para(
        "We resolved city naming mismatches between the general CPI and bottom-30% tables, normalized local basket weights to sum to 100% per province, and computed lag features using SQL window functions. Each province-month row carries 55 engineered attributes, including 1-month to 12-month inflation lags, rolling averages, lagged island wholesale growth, and retail rice changes. The cleaned master panel has 10,706 rows, with 9,494 valid year-over-year rows from January 2019 to October 2026. Missing values are transparent and bounded: 1,212 rows lack 2018 year-over-year baselines, 1,313 rows lack October 2026 wholesale lags, and 212 rows lack retail rice quotes in two island units."
    )

    b.add_para(
        "DuckDB exports two ZSTD-compressed Parquet marts totaling just 2.2 MB: a 521 KB provincial ML panel and a 1.65 MB division table. Inside the Streamlit app, every chart queries DuckDB with SQL aggregation so strictly fewer than 1,000 rows ever reach the browser. Fig. 2 maps this five-stage workflow."
    )

    b.add_fullwidth_figure(
        pipe_fig,
        "<b>Fig. 2.</b> End-to-end ETL, storage, modeling, and live serving architecture compressing 73.87 MB of raw PSA tables into 2.2 MB of Parquet marts.",
        width_in=6.75,
    )

    b.add_sub_heading("C. Chronological Forecasting and Evaluation Setup")
    b.add_para(
        "Random train-test splits leak future inflation regimes into training data and create misleadingly high scores. We therefore enforce strict time order. Every machine learning model trains on January 2020 through December 2024 and predicts unseen months from January 2025 through September 2026, totaling 2,121 out-of-time provincial test observations. We also run expanding TimeSeriesSplit cross-validation inside the training window."
    )

    b.add_para(
        "Two prediction tasks run side by side. First, a regression task predicts next-month provincial inflation using Ridge Regression, Random Forest, and XGBoost [3], benchmarked against a naive persistence baseline and national SARIMAX [4]. Second, a classification task predicts whether a province will breach the 4.0% BSP target next month using Logistic Regression, Random Forest, and XGBoost. To show forecast risk honestly, Gradient Boosting quantile models compute 80% and 95% prediction bands alongside SARIMAX confidence intervals, and exact TreeSHAP values explain feature impact [5]."
    )

    # ------------------------------------------------------------------
    # SECTION III
    # ------------------------------------------------------------------
    b.add_section_heading("III. Visualization Design Rationale and Interaction Architecture")

    b.add_sub_heading("A. Layered Portfolio: Five Basic, Five Advanced, and Predictive Views")
    b.add_para(
        "Our dashboard follows a clear three-step reading path so both executives and technical auditors can navigate it easily. Every plot uses colorblind-safe Viridis and Safe palettes, clear axis units, and interactive hover tooltips [6]."
    )

    b.add_bullets([
        "<b>Tab 1 Basic EDA:</b> Pairs national headline, core, bottom-30%, and rice trend lines with a pre-aggregated inflation histogram, regional median bars shaded by low-income gap, rice-to-headline scatter analysis, and a driver correlation summary. Statistical test badges display verified ANOVA and paired t-test metrics.",
        "<b>Tab 2 Five Advanced Views:</b> Displays a 101-unit Philippine bubble map of mean inflation and breach frequency, an interactive sunburst from island to region to spending group, a Sankey transmission flow chart, a six-axis parallel coordinates plot across provinces, and a stacked streamgraph of category drivers over time.",
        "<b>Tab 3 Forecasting and XAI:</b> Combines the out-of-time model leaderboard, a live SARIMAX fan chart with shaded 80% and 95% confidence bands, a TreeSHAP driver chart, an interactive ROC curve with live confusion matrix counts, and a cross-filtered high-error province trace.",
        "<b>Tab 4 What-If Simulator:</b> Lets planners pick any province or city and adjust rice, utility, fuel, and wholesale shocks to see live forecast shifts and low-income impact.",
    ])

    b.add_fullwidth_figure(
        FIG_DIR / "fig07_advanced_sankey_transmission.png",
        "<b>Fig. 3.</b> Sankey flow tracing 2022 to 2026 inflation pressure from broad supply channels through household basket divisions into Luzon, Visayas, and Mindanao.",
        width_in=6.20,
    )

    b.add_sub_heading("B. Why Each Advanced Visual Fits the Story")
    b.add_para(
        "Each advanced chart answers a question that simple bar charts miss. The geospatial map reveals that inflation hotspots cluster in outer island provinces like Batanes, Dinagat Islands, and Mindoro rather than inside Metro Manila. The Sankey chart in Fig. 3 proves at a glance that food and non-alcoholic beverages drive 48% of recent price pressure at 2.14 percentage points, followed by housing, utilities, and transport at 28% or 1.23 percentage points. Parallel coordinates let analysts spot provinces where food, rice, and utilities spike together, while the streamgraph tracks how food dominated the 2023 surge and returned sharply alongside transport in 2026."
    )

    b.add_sub_heading("C. Interactive Controls, Cross-Filtering, and SQL Transparency")
    b.add_para(
        "Global sidebar filters let users slice by island cluster, region, and year range, updating every KPI card and chart instantly through parameterized DuckDB SQL queries. An expandable SQL Audit Trail panel exposes the exact query running under the hood so instructors and auditors can verify the under-1,000-row aggregation rule."
    )

    b.add_para(
        "To connect diagnostics back to raw provincial behavior, Tab 3 includes two synchronization controls. Moving the sidebar breach threshold slider updates the live precision, recall, and confusion matrix counts on the 2025 to 2026 holdout set. Below the ROC chart, a High-Error Holdout Inspector ranks the ten largest prediction misses and lets the user pick any province to plot its observed versus predicted curve with high-error months flagged in red. Meanwhile, a forecast horizon slider dynamically retrains SARIMAX from 3 to 12 months ahead for any chosen region."
    )

    # ------------------------------------------------------------------
    # SECTION IV
    # ------------------------------------------------------------------
    b.add_section_heading("IV. Policy and Operational Insights Derived from Government Data")

    b.add_sub_heading("A. Provincial Price Gaps and Low-Income Vulnerability")
    b.add_para(
        "Statistical tests across all 9,494 provincial observations confirm that inflation is not uniform nationwide. As illustrated in Fig. 4, monthly headline inflation differs significantly across 18 regions with one-way ANOVA F = 5.53 and p < 0.001, and island groups also diverge with F = 8.22 and p < 0.001. Region V Bicol averaged 4.39% and broke the 4.0% BSP target 53.6% of the time. MIMAROPA followed at 4.30% with 52.5% breach frequency. At the other end, NCR averaged 3.41% and breached 4.0% only 29.8% of the time. Remote provinces face the steepest averages: Batanes reached 5.25%, Dinagat Islands hit 5.07%, and Occidental Mindoro exceeded 4.0% in 64.9% of all months."
    )

    b.add_para(
        "More importantly, food inflation hurts poorer families much faster than average shoppers. Across the full period, bottom-30% households experienced 0.16 percentage points higher inflation than all-income households with paired t = 11.83 and p < 0.001. During months when food inflation exceeded 5%, that gap jumped nearly sixfold to 0.95 percentage points with paired t = 43.52. The reason is simple basket math. In City of Baguio, rice accounts for 4.50% of the general household basket, but 12.15% of the bottom-30% basket. In October 2026, when Baguio rice inflation hit 16.5%, local low-income inflation climbed to 9.5% against 9.3% overall. Upstream links also hold clearly: lagged wholesale inflation correlates at r = 0.43 with next-month provincial CPI, and local retail rice growth correlates at r = 0.45."
    )

    b.add_fullwidth_figure(
        FIG_DIR / "fig03_regional_anova_boxplot.png",
        "<b>Fig. 4.</b> Distribution of monthly provincial headline inflation across all 18 Philippine regions from 2019 to 2026 against the 2.0% and 4.0% BSP target lines.",
        width_in=6.30,
    )

    b.add_sub_heading("B. Honest Model Benchmarks and Why Linear Shrinkage Wins")
    b.add_para(
        "Table II summarizes out-of-time accuracy across the turbulent 2025 to 2026 holdout period. For next-month numerical inflation, Ridge Regression achieves the lowest error at 0.93 percentage points MAE and R² of 0.874, outperforming Random Forest at 1.05, XGBoost at 1.16, and national SARIMAX at 2.20. Tree ensembles scored slightly better inside 2020 to 2024 cross-validation, yet struggled out of time because 2025 dropped to 0.9% before 2026 surged back to 7.8%. Tree leaves clip near past training ranges, whereas Ridge extrapolates smoothly across turning points. For the same reason, quantile 80% and 95% bands in Fig. 5 achieved 62.9% and 75.5% empirical coverage during the rapid 2026 jump, highlighting why fan charts must communicate widening risk clearly."
    )

    b.add_table(
        label="TABLE II",
        title="Out-of-Time 2025 to 2026 Holdout Model Benchmarks",
        headers=["Model and Task", "MAE / Acc", "RMSE / F1", "R² / AUC"],
        rows=[
            ("BANNER", "Next-Month YoY Regression"),
            ["<b>Ridge Regression</b>", "<b>0.93 pp</b>", "<b>1.30 pp</b>", "<b>0.874</b>"],
            ["Persistence Baseline", "0.93 pp", "1.34 pp", "0.867"],
            ["Random Forest Regressor", "1.05 pp", "1.42 pp", "0.849"],
            ["XGBoost Regressor", "1.16 pp", "1.54 pp", "0.824"],
            ["National SARIMAX", "2.20 pp", "2.56 pp", "N/A"],
            ("BANNER", "BSP >4.0% Breach Classification"),
            ["<b>XGBoost Classifier</b>", "<b>92.6%</b>", "<b>0.897</b>", "<b>0.976</b>"],
            ["Logistic Regression", "92.4%", "0.893", "0.980"],
            ["Random Forest Classifier", "92.3%", "0.892", "0.965"],
        ],
        col_widths=[1.54, 0.64, 0.64, 0.64],
        right_cols={1, 2, 3},
        font_size=7.7,
        keep_together=True,
    )

    b.add_para(
        "We also report an honest forecasting insight shown in Fig. 6. Because year-over-year inflation overlaps eleven of twelve months with the previous month, current headline inflation dominates TreeSHAP at 2.50 percentage points mean impact, followed by three-month momentum, core inflation, seasonality, wholesale lag, and fuel costs. As a result, Ridge beats naive persistence only narrowly on one-month point error. The real decision value therefore lies in three operational outputs: the 92.6% accurate early-warning breach classifier with 0.976 ROC-AUC, the 6-month SARIMAX outlook projecting national inflation near 8.0% through February 2027 before easing toward 3.4% by April 2027, and the basket-weighted shock simulator."
    )

    b.add_fullwidth_figure(
        FIG_DIR / "fig11_predictive_fan_chart_sarimax_xgb.png",
        "<b>Fig. 5.</b> Out-of-time provincial forecast tracking across 2025 to 2026 with 80% and 95% quantile bands, followed by the 6-month national SARIMAX fan chart.",
        width_in=6.45,
    )

    b.add_fullwidth_figure(
        FIG_DIR / "fig12_xai_treeshap_and_roc_breach.png",
        "<b>Fig. 6.</b> Left: Exact TreeSHAP drivers for next-month provincial inflation. Right: Holdout ROC curves for provincial breaches above the 4.0% BSP target.",
        width_in=6.45,
    )

    b.add_sub_heading("C. Live Supply-Shock Simulation and Agency Action Plan")
    b.add_para(
        "Table III shows how the live Tab 4 simulator responds when we inject the exact same supply shock starting from October 2026 data: +8.0 percentage points in rice, +5.0 in electricity and LPG, +6.0 in diesel and transport fuel, and +3.0 in wholesale spillover. Even with identical shock inputs, Camarines Sur absorbs nearly double the direct basket impact of Baguio City because rice carries a 12.57% share in Camarines Sur versus 4.50% in Baguio. Meanwhile, Baguio low-income families absorb an extra 0.61 percentage points from rice alone compared with general Baguio households, proving that a single national relief formula misses local needs."
    )

    b.add_table(
        label="TABLE III",
        title="Simulated Nov 2026 Shock: +8pp Rice, +5pp Util, +6pp Fuel, +3pp GWPI",
        headers=["Area", "Oct Obs", "Base", "Shock", "Direct", "B30 Extra"],
        rows=[
            ["City of Baguio", "9.30%", "8.95%", "9.62%", "+0.77 pp", "+0.61 pp"],
            ["Camarines Sur", "10.10%", "9.70%", "10.93%", "+1.45 pp", "+0.68 pp"],
            ["NCR", "5.90%", "6.05%", "6.80%", "+0.87 pp", "+0.41 pp"],
        ],
        col_widths=[1.02, 0.48, 0.45, 0.48, 0.51, 0.52],
        right_cols={1, 2, 3, 4, 5},
        font_size=7.6,
        keep_together=True,
    )

    b.add_para(
        "These empirical patterns translate into three straightforward policy actions for government stakeholders:"
    )

    b.add_bullets([
        "<b>Targeted Rice and Food Relief for DA and DTI:</b> Prioritize rice buffer deployment and Kadiwa rolling stores in high-weight, high-breach provinces such as Bicol, MIMAROPA, and island areas whenever local retail rice inflation crosses 8% rather than waiting for national averages to spike.",
        "<b>Regional Early-Warning Thresholds for BSP:</b> Use the tunable breach classifier in Tab 3 to trigger regional monitoring alerts when predicted probability crosses 0.45, capturing 85.6% of upcoming provincial target breaches with 94.2% precision.",
        "<b>Low-Income Cost Protection for NEDA and LGUs:</b> Anchor local wage reviews and targeted household subsidies to bottom-30% CPI baskets during food surges, since low-income urban families in cities like Baguio face nearly triple the rice weight of all-income indices.",
    ])

    # ------------------------------------------------------------------
    # SECTION V
    # ------------------------------------------------------------------
    b.add_section_heading("V. Conclusion and Limitations")

    b.add_para(
        "PesoPulse demonstrates how open Philippine government statistics can move from static spreadsheets into a fast, transparent decision-support tool. By processing 4.5 million PSA price cells inside DuckDB and serving compact Parquet marts through Streamlit, our platform delivers five basic views, five advanced visual encodings, honest out-of-time forecasts, explainable drivers, and real-time supply-shock simulation at minimal compute cost."
    )

    b.add_para(
        "Two practical limitations guide future upgrades. First, PSA OpenSTAT does not publish province-level monthly rainfall, typhoon disruption logs, or port freight rates, which explain sudden spikes in island provinces like Batanes and Mindoro. Second, one-month year-over-year CPI moves closely with recent history, so future work should add month-on-month food price forecasting and direct PAGASA weather feeds to extend early warning from one month to a full quarter."
    )

    b.add_para(
        "The full codebase, reproducible DuckDB scripts, compact Parquet marts, and Streamlit application are maintained under version control with distributed commits across all three team members covering data engineering, visual analytics, and predictive deployment. Because the runtime files weigh only 2.2 MB, the platform deploys cleanly to Streamlit Community Cloud without external database servers."
    )

    # ------------------------------------------------------------------
    # REFERENCES
    # ------------------------------------------------------------------
    b.add_references([
        "Philippine Statistics Authority, \"Consumer Price Index for All Income Households, Bottom 30% Income Households, Purchasing Power of the Peso, General Wholesale Price Index, and Retail Rice Prices 2018 to 2026,\" OpenSTAT Portal, 2026. Available: https://openstat.psa.gov.ph",
        "M. Raasveldt and H. Muhleisen, \"DuckDB: An Embeddable Analytical Database,\" in Proc. ACM SIGMOD Int. Conf. Management of Data, Amsterdam, Netherlands, 2019, pp. 1981-1984.",
        "T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785-794.",
        "G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, Time Series Analysis: Forecasting and Control, 5th ed. Hoboken, NJ, USA: Wiley, 2015.",
        "S. M. Lundberg et al., \"From Local Explanations to Global Understanding with Explainable AI for Trees,\" Nature Machine Intelligence, vol. 2, no. 1, pp. 56-67, Jan. 2020.",
        "T. Munzner, Visualization Analysis and Design. Boca Raton, FL, USA: CRC Press, 2014.",
        "Bangko Sentral ng Pilipinas, \"Inflation Targeting and Monetary Policy Framework in the Philippines,\" BSP Technical Primer, Manila, Philippines, 2025.",
    ])

    # Trailing continuous break balances final two columns evenly on Page 6
    b.switch_columns(1)

    docx_out = DOCS_DIR / "IEEE_Technical_Project_Paper_6Pages.docx"
    md_out = DOCS_DIR / "IEEE_Technical_Project_Paper_6Pages.md"
    b.save_and_audit(docx_out, md_out)


if __name__ == "__main__":
    main()
