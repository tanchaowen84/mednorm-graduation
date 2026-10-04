from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "论文正文.md"
OUTPUT = ROOT / "张艺馨毕业论文初稿_大规模医学知识图谱的疾病实体标准化方法_v1.docx"
ASSET_DIR = ROOT / "assets"


# Base preset: narrative_proposal.
# Named override "Chinese academic thesis": A4 page, Chinese thesis typography,
# black headings, 1.5-line body text and the editorial_cover opening pattern.
TOKENS = {
    "page_width_cm": 21.0,
    "page_height_cm": 29.7,
    "margin_top_cm": 2.5,
    "margin_bottom_cm": 2.5,
    "margin_left_cm": 3.0,
    "margin_right_cm": 2.5,
    "header_cm": 1.5,
    "footer_cm": 1.5,
    "content_width_dxa": 8787,
    "table_indent_dxa": 120,
    "body_east_asia": "Noto Serif CJK SC",
    "body_ascii": "Noto Serif CJK SC",
    "heading_east_asia": "Noto Sans CJK SC",
    "heading_ascii": "Noto Sans CJK SC",
    "body_size_pt": 12,
    "body_after_pt": 0,
    "body_line_spacing": 1.5,
    "h1_size_pt": 16,
    "h1_before_pt": 0,
    "h1_after_pt": 18,
    "h2_size_pt": 14,
    "h2_before_pt": 12,
    "h2_after_pt": 6,
    "h3_size_pt": 12,
    "h3_before_pt": 8,
    "h3_after_pt": 4,
    "table_header_fill": "F4F6F9",
    "table_cell_margins": {"top": 80, "bottom": 80, "start": 120, "end": 120},
}


def set_run_font(
    run,
    *,
    east_asia: str | None = None,
    ascii_font: str | None = None,
    size: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    color: str | None = None,
) -> None:
    requested_font = east_asia or TOKENS["body_east_asia"]
    if requested_font in {"Heiti SC", "STFangsong", "Menlo", TOKENS["heading_east_asia"]}:
        east_asia = TOKENS["heading_east_asia"]
    else:
        east_asia = TOKENS["body_east_asia"]
    ascii_font = east_asia
    # LibreOffice's DOCX converter can ignore the eastAsia fallback when the
    # ASCII font lacks CJK glyphs. Use the CJK family for every rFonts slot;
    # Songti and Heiti both contain Latin glyphs and render consistently in
    # Word and in the headless verification pipeline.
    run.font.name = east_asia
    rfonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), east_asia)
    rfonts.set(qn("w:hAnsi"), east_asia)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:cs"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in TOKENS["table_cell_margins"].items():
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TOKENS["table_indent_dxa"]))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        for cell, width in zip(row.cells, widths_dxa, strict=True):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table, color: str = "B7BEC8", size: str = "4") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_section_geometry(section) -> None:
    section.page_width = Cm(TOKENS["page_width_cm"])
    section.page_height = Cm(TOKENS["page_height_cm"])
    section.top_margin = Cm(TOKENS["margin_top_cm"])
    section.bottom_margin = Cm(TOKENS["margin_bottom_cm"])
    section.left_margin = Cm(TOKENS["margin_left_cm"])
    section.right_margin = Cm(TOKENS["margin_right_cm"])
    section.header_distance = Cm(TOKENS["header_cm"])
    section.footer_distance = Cm(TOKENS["footer_cm"])


def set_page_number_format(section, *, fmt: str, start: int) -> None:
    sect_pr = section._sectPr
    pg_num = sect_pr.find(qn("w:pgNumType"))
    if pg_num is None:
        pg_num = OxmlElement("w:pgNumType")
        sect_pr.append(pg_num)
    pg_num.set(qn("w:fmt"), fmt)
    pg_num.set(qn("w:start"), str(start))


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_begin, instr, fld_sep, fld_end])
    set_run_font(run, size=9, color="666666")


def configure_header_footer(section, *, show_header: bool = True) -> None:
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    header = section.header
    header_p = header.paragraphs[0]
    header_p.clear()
    if show_header:
        header_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = header_p.add_run("大规模医学知识图谱的疾病实体标准化方法")
        set_run_font(run, east_asia="Songti SC", size=9, color="666666")
    footer_p = section.footer.paragraphs[0]
    footer_p.clear()
    add_page_field(footer_p)


def configure_styles(doc: Document) -> None:
    styles = doc.styles

    normal = styles["Normal"]
    normal.font.name = TOKENS["body_ascii"]
    normal.font.size = Pt(TOKENS["body_size_pt"])
    normal._element.rPr.rFonts.set(qn("w:ascii"), TOKENS["body_ascii"])
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), TOKENS["body_ascii"])
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["body_east_asia"])
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Cm(0.84)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(TOKENS["body_after_pt"])
    normal.paragraph_format.line_spacing = TOKENS["body_line_spacing"]
    normal.paragraph_format.widow_control = True

    for name, level in (("Heading 1", 1), ("Heading 2", 2), ("Heading 3", 3)):
        style = styles[name]
        style.font.name = TOKENS["heading_ascii"]
        style._element.rPr.rFonts.set(qn("w:ascii"), TOKENS["heading_ascii"])
        style._element.rPr.rFonts.set(qn("w:hAnsi"), TOKENS["heading_ascii"])
        style._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["heading_east_asia"])
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.first_line_indent = Cm(0)
        if level == 1:
            style.font.size = Pt(TOKENS["h1_size_pt"])
            style.paragraph_format.space_before = Pt(TOKENS["h1_before_pt"])
            style.paragraph_format.space_after = Pt(TOKENS["h1_after_pt"])
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style.paragraph_format.page_break_before = True
        elif level == 2:
            style.font.size = Pt(TOKENS["h2_size_pt"])
            style.paragraph_format.space_before = Pt(TOKENS["h2_before_pt"])
            style.paragraph_format.space_after = Pt(TOKENS["h2_after_pt"])
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        else:
            style.font.size = Pt(TOKENS["h3_size_pt"])
            style.paragraph_format.space_before = Pt(TOKENS["h3_before_pt"])
            style.paragraph_format.space_after = Pt(TOKENS["h3_after_pt"])
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    caption = styles["Caption"]
    caption.font.name = TOKENS["body_ascii"]
    caption.font.size = Pt(10.5)
    caption.font.color.rgb = RGBColor(0, 0, 0)
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["body_east_asia"])
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.first_line_indent = Cm(0)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Number"):
        style = styles[style_name]
        style.font.name = TOKENS["body_ascii"]
        style.font.size = Pt(11.5)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), TOKENS["body_east_asia"])
        style.paragraph_format.left_indent = Cm(0.95)
        style.paragraph_format.first_line_indent = Cm(-0.49)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.208


def add_rich_text(paragraph, text: str, *, size: float | None = None) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`)")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            set_run_font(run, size=size)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size=size, bold=True)
        else:
            run = paragraph.add_run(token[1:-1])
            set_run_font(
                run,
                east_asia="STFangsong",
                ascii_font="Menlo",
                size=size or 10,
                color="333333",
            )
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        set_run_font(run, size=size)


def add_cover(doc: Document) -> None:
    section = doc.sections[0]
    set_section_geometry(section)
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    section.header.paragraphs[0].clear()
    section.footer.paragraphs[0].clear()

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(26)
    r = p.add_run("本科毕业论文")
    set_run_font(r, east_asia="Heiti SC", size=18, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(72)
    p.paragraph_format.line_spacing = 1.35
    r = p.add_run("大规模医学知识图谱的\n疾病实体标准化方法")
    set_run_font(r, east_asia="Heiti SC", size=24, bold=True)

    rows = [
        ("学生姓名", "张艺馨"),
        ("学　　号", "**********"),
        ("专　　业", "计算机科学与技术"),
        ("指导教师", "陈清彩"),
    ]
    table = doc.add_table(rows=len(rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, (label, value) in enumerate(rows):
        table.cell(i, 0).width = Cm(3.2)
        table.cell(i, 1).width = Cm(6.4)
        for j, text in enumerate((label, value)):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(text)
            set_run_font(run, east_asia="Songti SC", size=14)
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                node = OxmlElement(f"w:{edge}")
                node.set(qn("w:val"), "nil")
                borders.append(node)
            tc_pr.append(borders)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(46)
    r = p.add_run("2026年8月")
    set_run_font(r, east_asia="Songti SC", size=14)


def add_toc(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)
    run = title.add_run("目　录")
    set_run_font(run, east_asia="Heiti SC", size=16, bold=True)

    entries = [
        ("摘要", "i", 0),
        ("Abstract", "iii", 0),
        ("第1章 绪论", "1", 0),
        ("1.1 研究背景", "1", 1),
        ("1.2 问题定义与边界", "2", 1),
        ("1.3 主要难点", "2", 1),
        ("1.4 本文的主要工作", "3", 1),
        ("1.5 论文结构", "4", 1),
        ("第2章 相关研究与技术基础", "5", 0),
        ("2.1 医学实体标准化", "5", 1),
        ("2.2 表示学习与同义词对齐", "5", 1),
        ("2.3 中文医学标准化与多含义问题", "6", 1),
        ("2.4 候选召回与精排", "7", 1),
        ("2.5 知识图谱在标准化中的作用", "7", 1),
        ("2.6 现有方法的不足与本文思路", "8", 1),
        ("第3章 数据、任务建模与评价方法", "10", 0),
        ("3.1 数据来源", "10", 1),
        ("3.2 ICD候选词表构建", "11", 1),
        ("3.3 CPubMed-KG清洗与ICD对齐", "11", 1),
        ("3.4 数据特征与覆盖率", "12", 1),
        ("3.5 输入预处理", "13", 1),
        ("3.6 评价指标", "13", 1),
        ("3.7 数据划分与防泄漏策略", "14", 1),
        ("第4章 大规模医学知识图谱增强的疾病实体标准化方法", "15", 0),
        ("4.1 总体思路", "15", 1),
        ("4.2 稀疏候选召回", "16", 1),
        ("4.3 BGE稠密召回与RRF融合", "17", 1),
        ("4.4 训练样本原型证据", "18", 1),
        ("4.5 标签数量预测", "19", 1),
        ("4.6 普通MacBERT候选精排", "19", 1),
        ("4.7 知识图谱候选画像精排", "20", 1),
        ("4.8 BGE候选重排", "21", 1),
        ("4.9 Qwen结构拆分信号", "23", 1),
        ("4.10 五路得分融合", "23", 1),
        ("4.11 分组阈值解码", "24", 1),
        ("4.12 训练与推理算法", "25", 1),
        ("4.13 可复现配置与版本固定", "26", 1),
        ("第5章 实验结果与分析", "28", 0),
        ("5.1 实验目的", "28", 1),
        ("5.2 实验设置", "28", 1),
        ("5.3 候选召回结果", "29", 1),
        ("5.4 主实验结果", "31", 1),
        ("5.5 随机种子稳定性", "32", 1),
        ("5.6 固定参数消融分析", "33", 1),
        ("5.7 分组结果", "34", 1),
        ("5.8 典型案例分析", "36", 1),
        ("5.9 结果讨论", "37", 1),
        ("第6章 系统实现与复现说明", "39", 0),
        ("6.1 工程结构", "39", 1),
        ("6.2 运行环境与资源消耗", "39", 1),
        ("6.3 从原始数据到最终结果的复现流程", "40", 1),
        ("6.4 模型与结果保存", "41", 1),
        ("6.5 推理接口设计", "42", 1),
        ("6.6 复现检查点", "42", 1),
        ("第7章 总结与展望", "44", 0),
        ("7.1 工作总结", "44", 1),
        ("7.2 方法的实际价值", "45", 1),
        ("7.3 现有不足", "45", 1),
        ("7.4 后续研究方向", "46", 1),
        ("参考文献", "47", 0),
        ("致谢", "51", 0),
        ("附录A 关键复现参数", "52", 0),
        ("A.1 固定候选和召回参数", "52", 1),
        ("A.2 图谱增强参数", "52", 1),
        ("A.3 最终融合与阈值", "53", 1),
        ("附录B Qwen结构拆分约束", "54", 0),
        ("附录C 复现文件清单", "55", 0),
    ]
    for entry_title, page, level in entries:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.left_indent = Cm(0.74 if level else 0)
        p.paragraph_format.space_before = Pt(3 if not level else 0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p_pr = p._p.get_or_add_pPr()
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "right")
        tab.set(qn("w:leader"), "dot")
        tab.set(qn("w:pos"), str(TOKENS["content_width_dxa"] - (420 if level else 0)))
        tabs.append(tab)
        p_pr.append(tabs)
        title_run = p.add_run(entry_title)
        set_run_font(title_run, size=10.5, bold=not level)
        tab_run = p.add_run("\t")
        set_run_font(tab_run, size=10.5)
        page_run = p.add_run(page)
        set_run_font(page_run, size=10.5, bold=not level)


def add_equation(doc: Document, value: str, number: str) -> None:
    table = doc.add_table(rows=1, cols=2)
    widths = [int(TOKENS["content_width_dxa"] * 0.88), int(TOKENS["content_width_dxa"] * 0.12)]
    widths[-1] = TOKENS["content_width_dxa"] - widths[0]
    set_table_geometry(table, widths)
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            node = OxmlElement(f"w:{edge}")
            node.set(qn("w:val"), "nil")
            borders.append(node)
        tc_pr.append(borders)
    left, right = table.rows[0].cells
    p = left.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    add_rich_text(p, value, size=11.5)
    p = right.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.first_line_indent = Cm(0)
    add_rich_text(p, f"({number})", size=11.5)


def add_markdown_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    cols = len(rows[0])
    max_lengths = [max(4, max(len(row[i]) for row in rows)) for i in range(cols)]
    total = sum(max_lengths)
    widths = [max(800, int(TOKENS["content_width_dxa"] * x / total)) for x in max_lengths]
    delta = TOKENS["content_width_dxa"] - sum(widths)
    widths[-1] += delta
    if widths[-1] < 800:
        take = 800 - widths[-1]
        widths[-1] = 800
        widths[0] -= take

    table = doc.add_table(rows=len(rows), cols=cols)
    set_table_geometry(table, widths)
    set_table_borders(table)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            if i == 0:
                set_cell_shading(cell, TOKENS["table_header_fill"])
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            add_rich_text(p, value, size=9.5)
            for run in p.runs:
                run.bold = i == 0
        tr_pr = table.rows[i]._tr.get_or_add_trPr()
        if i == 0:
            repeat_header = OxmlElement("w:tblHeader")
            repeat_header.set(qn("w:val"), "true")
            tr_pr.append(repeat_header)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def generate_figures() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    regular_path = "/Library/Fonts/Arial Unicode.ttf"
    bold_path = "/System/Library/Fonts/STHeiti Light.ttc"

    def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(bold_path if bold else regular_path, size=size)

    def centered_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, text_font: ImageFont.FreeTypeFont, fill: str = "#1F2933") -> None:
        left, top, right, bottom = box
        bounds = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=7, align="center")
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        draw.multiline_text(((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]), value, font=text_font, fill=fill, spacing=7, align="center")

    def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#66717E") -> None:
        draw.line([start, end], fill=fill, width=5)
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = (ex, ey)
        base = (ex - ux * 24, ey - uy * 24)
        points = [tip, (base[0] + px * 11, base[1] + py * 11), (base[0] - px * 11, base[1] - py * 11)]
        draw.polygon(points, fill=fill)

    # Method pipeline.
    canvas = Image.new("RGB", (2160, 1296), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(48, bold=True)
    box_font = font(30)
    title = "整体方法流程"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((2160 - (title_box[2] - title_box[0])) / 2, 55), title, font=title_font, fill="#17202A")
    boxes = [
        ((70, 235, 430, 400), "原始诊断短语\n字符清洗", "#EAF2F8"),
        ((540, 235, 940, 400), "候选召回\n稀疏 + BGE + RRF", "#E8F6F3"),
        ((1050, 235, 1425, 400), "Top-400\nICD候选集合", "#FCF3CF"),
        ((1535, 235, 2080, 400), "多路精排\nMacBERT / KG画像 / BGE", "#F5EEF8"),
        ((150, 650, 580, 815), "训练样本原型证据\n相似病例标签记忆", "#FDEDEC"),
        ((700, 650, 1110, 815), "标签数量预测\n1 / 2 / 3+", "#EBF5FB"),
        ((1260, 650, 1660, 815), "Qwen结构拆分\n仅产生激活信号", "#F4ECF7"),
        ((710, 1030, 1490, 1210), "五路得分融合 + 分组阈值解码\n输出ICD标准疾病实体", "#D5F5E3"),
    ]
    for box, value, fill in boxes:
        draw.rounded_rectangle(box, radius=24, fill=fill, outline="#58636F", width=4)
        centered_multiline(draw, box, value, box_font)
    for start, end in [
        ((430, 318), (540, 318)),
        ((940, 318), (1050, 318)),
        ((1425, 318), (1535, 318)),
        ((1238, 400), (1050, 1030)),
        ((1805, 400), (1360, 1030)),
        ((365, 815), (820, 1030)),
        ((905, 815), (980, 1030)),
        ((1460, 815), (1240, 1030)),
    ]:
        arrow(draw, start, end)
    canvas.save(ASSET_DIR / "method_pipeline.png", dpi=(180, 180))

    # Retrieval line chart.
    canvas = Image.new("RGB", (1692, 972), "white")
    draw = ImageDraw.Draw(canvas)
    axis_font = font(26)
    small_font = font(23)
    legend_font = font(25)
    left, top, right, bottom = 150, 85, 1600, 805
    y_min, y_max = 65.0, 96.0
    ks = [50, 100, 200, 400, 800]
    series = [
        ("字符稀疏召回", [69.68, 76.40, 81.35, 85.92, 88.61], "#4C78A8"),
        ("BGE稠密召回", [84.07, 88.81, 91.27, 92.38, 93.26], "#E15759"),
        ("RRF融合召回", [85.69, 89.46, 91.75, 92.66, 93.26], "#59A14F"),
    ]
    def x_pos(idx: int) -> float:
        return left + idx * (right - left) / (len(ks) - 1)
    def y_pos(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
    for tick in [65, 70, 75, 80, 85, 90, 95]:
        y = y_pos(tick)
        draw.line([(left, y), (right, y)], fill="#D9DEE5", width=2)
        label = str(tick)
        bounds = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - 22 - (bounds[2] - bounds[0]), y - 14), label, font=small_font, fill="#3C4651")
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="#444C56", width=4)
    for idx, value in enumerate(ks):
        x = x_pos(idx)
        draw.line([(x, bottom), (x, bottom + 10)], fill="#444C56", width=3)
        label = str(value)
        bounds = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (bounds[2] - bounds[0]) / 2, bottom + 18), label, font=small_font, fill="#3C4651")
    for name, values, color in series:
        points = [(x_pos(i), y_pos(v)) for i, v in enumerate(values)]
        draw.line(points, fill=color, width=7, joint="curve")
        for x, y in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="white", outline=color, width=6)
    draw.text((730, 895), "候选数量 K", font=axis_font, fill="#26313C")
    y_label = Image.new("RGBA", (300, 80), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "标签召回率（%）", font=axis_font, fill="#26313C")
    y_label = y_label.rotate(90, expand=True)
    canvas.paste(y_label, (25, 325), y_label)
    legend_x = 710
    for idx, (name, _, color) in enumerate(series):
        y = 115 + idx * 48
        draw.line([(legend_x, y + 15), (legend_x + 55, y + 15)], fill=color, width=7)
        draw.text((legend_x + 70, y), name, font=legend_font, fill="#26313C")
    canvas.save(ASSET_DIR / "retrieval_recall.png", dpi=(180, 180))

    # Segment bar chart.
    canvas = Image.new("RGB", (1692, 936), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 145, 70, 1600, 775
    names = ["总体", "ICD覆盖", "多实体", "单实体", "结构激活", "未激活"]
    values = [71.73, 75.08, 78.77, 55.63, 73.23, 69.94]
    colors = ["#4C78A8", "#72B7B2", "#59A14F", "#E15759", "#B279A2", "#9D755D"]
    y_min, y_max = 45.0, 84.0
    def bar_y(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
    for tick in [45, 50, 55, 60, 65, 70, 75, 80]:
        y = bar_y(tick)
        draw.line([(left, y), (right, y)], fill="#D9DEE5", width=2)
        bounds = draw.textbbox((0, 0), str(tick), font=small_font)
        draw.text((left - 22 - (bounds[2] - bounds[0]), y - 14), str(tick), font=small_font, fill="#3C4651")
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="#444C56", width=4)
    slot = (right - left) / len(names)
    for idx, (name, value, color) in enumerate(zip(names, values, colors, strict=True)):
        center = left + slot * (idx + 0.5)
        width = slot * 0.6
        y = bar_y(value)
        draw.rectangle((center - width / 2, y, center + width / 2, bottom), fill=color)
        value_text = f"{value:.2f}"
        bounds = draw.textbbox((0, 0), value_text, font=small_font)
        draw.text((center - (bounds[2] - bounds[0]) / 2, y - 40), value_text, font=small_font, fill="#26313C")
        bounds = draw.textbbox((0, 0), name, font=small_font)
        draw.text((center - (bounds[2] - bounds[0]) / 2, bottom + 20), name, font=small_font, fill="#26313C")
    y_label = Image.new("RGBA", (280, 80), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Micro-F1（%）", font=axis_font, fill="#26313C")
    y_label = y_label.rotate(90, expand=True)
    canvas.paste(y_label, (20, 300), y_label)
    canvas.save(ASSET_DIR / "segment_performance.png", dpi=(180, 180))


def parse_markdown(doc: Document, text: str) -> None:
    lines = text.splitlines()
    i = 0
    frontmatter = True
    in_references = False
    body_section_started = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            parsed = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(parsed) >= 2 and all(set(cell) <= {"-", ":", " "} for cell in parsed[1]):
                parsed.pop(1)
            add_markdown_table(doc, parsed)
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.right_indent = Cm(0.5)
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run("\n".join(code))
            set_run_font(run, east_asia="STFangsong", ascii_font="Menlo", size=9.2, color="222222")
            p_pr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), "F5F6F7")
            p_pr.append(shd)
            continue

        if stripped == "[[TOC]]":
            add_toc(doc)
            i += 1
            continue

        if stripped == "[[PAGEBREAK]]":
            p = doc.add_paragraph()
            p.add_run().add_break(WD_BREAK.PAGE)
            i += 1
            continue

        match = re.fullmatch(r"\[\[TABLECAP:(.+)\]\]", stripped)
        if match:
            p = doc.add_paragraph(style="Caption")
            add_rich_text(p, match.group(1), size=10.5)
            i += 1
            continue

        match = re.fullmatch(r"\[\[FIG:([^|]+)\|(.+)\]\]", stripped)
        if match:
            path = ASSET_DIR / match.group(1)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run()
            run.add_picture(str(path), width=Inches(5.95))
            cap = doc.add_paragraph(style="Caption")
            add_rich_text(cap, match.group(2), size=10.5)
            i += 1
            continue

        match = re.fullmatch(r"\[\[EQ:(.+)\|([^|]+)\]\]", stripped)
        if match:
            add_equation(doc, match.group(1), match.group(2))
            i += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title in {"摘要", "Abstract"} and frontmatter:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(18)
                p.paragraph_format.first_line_indent = Cm(0)
                r = p.add_run(title if title == "Abstract" else "摘　要")
                set_run_font(r, east_asia="Heiti SC", size=16, bold=True)
            else:
                if title.startswith("第1章") and not body_section_started:
                    section = doc.add_section(WD_SECTION.NEW_PAGE)
                    set_section_geometry(section)
                    configure_header_footer(section, show_header=True)
                    set_page_number_format(section, fmt="decimal", start=1)
                    body_section_started = True
                    frontmatter = False
                p = doc.add_paragraph(title, style="Heading 1")
                for r in p.runs:
                    set_run_font(r, east_asia=TOKENS["heading_east_asia"], ascii_font=TOKENS["heading_ascii"], size=TOKENS["h1_size_pt"], bold=True)
                if title == "参考文献":
                    in_references = True
                elif in_references:
                    in_references = False
            i += 1
            continue

        if stripped.startswith("## "):
            p = doc.add_paragraph(stripped[3:].strip(), style="Heading 2")
            for r in p.runs:
                set_run_font(r, east_asia=TOKENS["heading_east_asia"], ascii_font=TOKENS["heading_ascii"], size=TOKENS["h2_size_pt"], bold=True)
            i += 1
            continue

        if stripped.startswith("### "):
            p = doc.add_paragraph(stripped[4:].strip(), style="Heading 3")
            for r in p.runs:
                set_run_font(r, east_asia=TOKENS["heading_east_asia"], ascii_font=TOKENS["heading_ascii"], size=TOKENS["h3_size_pt"], bold=True)
            i += 1
            continue

        if re.match(r"^-\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.first_line_indent = Cm(-0.49)
            add_rich_text(p, re.sub(r"^-\s+", "", stripped), size=11.5)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.first_line_indent = Cm(-0.49)
            add_rich_text(p, re.sub(r"^\d+\.\s+", "", stripped), size=11.5)
            i += 1
            continue

        p = doc.add_paragraph()
        if stripped.startswith("关键词：") or stripped.startswith("Keywords:"):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(8)
        if in_references or re.match(r"^\[\d+\]", stripped):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(-0.74)
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.line_spacing = 1.25
            p.paragraph_format.space_after = Pt(4)
        add_rich_text(p, stripped)
        i += 1


def add_update_fields_setting(doc: Document) -> None:
    settings = doc.settings._element
    node = settings.find(qn("w:updateFields"))
    if node is None:
        node = OxmlElement("w:updateFields")
        settings.append(node)
    node.set(qn("w:val"), "true")


def build() -> Path:
    generate_figures()
    doc = Document()
    configure_styles(doc)
    add_cover(doc)

    front = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_geometry(front)
    configure_header_footer(front, show_header=False)
    set_page_number_format(front, fmt="lowerRoman", start=1)

    parse_markdown(doc, SOURCE.read_text(encoding="utf-8"))
    add_update_fields_setting(doc)

    for section in doc.sections:
        set_section_geometry(section)
    doc.core_properties.title = "大规模医学知识图谱的疾病实体标准化方法"
    doc.core_properties.author = "张艺馨"
    doc.core_properties.subject = "本科毕业论文初稿"
    doc.core_properties.keywords = "疾病实体标准化; CPubMed-KG; CHIP-CDN; 候选召回; 精排; 知识图谱"
    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
