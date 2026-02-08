import io
import itertools
import logging
import os
import re
import time
import unicodedata
from abc import ABC
from abc import abstractmethod
from multiprocessing import Process
from pathlib import Path

import freetype
import pymupdf
from bitstring import BitStream

from babeldoc.assets.embedding_assets_metadata import FONT_NAMES
from babeldoc.format.pdf.document_il import PdfOriginalPath
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.matrix_helper import matrix_to_bytes
from babeldoc.format.pdf.document_il.utils.zstd_helper import zstd_decompress
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)

SUBSET_FONT_STAGE_NAME = "Subset font"
SAVE_PDF_STAGE_NAME = "Save PDF"

class RenderUnit(ABC):
    def __init__(self, render_order: int, sub_render_order: int = 0, xobj_id: str | None = None):
        self.render_order, self.sub_render_order, self.xobj_id = render_order, sub_render_order, xobj_id
        if self.render_order is None: self.render_order = 9999999999999999
        if self.sub_render_order is None: self.sub_render_order = 9999999999999999
    @abstractmethod
    def render(self, draw_op: BitStream, context: "RenderContext") -> None: pass
    def get_sort_key(self) -> tuple[int, int]: return (self.render_order, self.sub_render_order)

class CharacterRenderUnit(RenderUnit):
    def __init__(self, char: il_version_1.PdfCharacter, render_order: int, sub_render_order: int = 0):
        super().__init__(render_order, sub_render_order, char.xobj_id)
        self.char = char
    def render(self, draw_op: BitStream, context: "RenderContext") -> None:
        char = self.char
        if char.char_unicode == "\n" or char.pdf_character_id is None: return
        char_size, font_id = char.pdf_style.font_size, char.pdf_style.font_id
        encoding_length_map = context.xobj_encoding_length_map.get(self.xobj_id, context.page_encoding_length_map)
        if context.check_font_exists:
            available = context.xobj_available_fonts.get(self.xobj_id, context.available_font_list)
            if font_id not in available: return
        draw_op.append(b"q ")
        context.pdf_creator.render_graphic_state(draw_op, char.pdf_style.graphic_state)
        if char.vertical: draw_op.append(f"BT /{font_id} {char_size:f} Tf 0 1 -1 0 {char.box.x2:f} {char.box.y:f} Tm ".encode())
        else: draw_op.append(f"BT /{font_id} {char_size:f} Tf 1 0 0 1 {char.box.x:f} {char.box.y:f} Tm ".encode())
        encoding_length = encoding_length_map.get(font_id, context.all_encoding_length_map.get(font_id))
        if encoding_length is None: return
        draw_op.append(f"<{char.pdf_character_id:0{encoding_length * 2}x}>".upper().encode() + b" Tj ET Q \n")

# ... (استعادة FormRenderUnit, RectangleRenderUnit, CurveRenderUnit, RenderContext وجميع دوال اليوني كود المساعدة كما هي في ملفك الأصلي) ...
class FormRenderUnit(RenderUnit):
    def __init__(self, form: il_version_1.PdfForm, render_order: int, sub_render_order: int = 0):
        super().__init__(render_order, sub_render_order, form.xobj_id); self.form = form
    def render(self, draw_op, context):
        form = self.form; draw_op.append(b"q ")
        if form.relocation_transform and len(form.relocation_transform) == 6:
            try: draw_op.append(matrix_to_bytes(tuple(float(x) for x in form.relocation_transform)))
            except: pass
        draw_op.append(matrix_to_bytes(form.pdf_matrix) + b" " + form.graphic_state.passthrough_per_char_instruction.encode() + b" ")
        if form.pdf_form_subtype.pdf_xobj_form: draw_op.append(f" /{form.pdf_form_subtype.pdf_xobj_form.do_args} Do ".encode())
        draw_op.append(b" Q\n")

class RectangleRenderUnit(RenderUnit):
    def __init__(self, rectangle, render_order, sub_render_order=0, line_width=0.4):
        super().__init__(render_order, sub_render_order, rectangle.xobj_id); self.rectangle, self.line_width = rectangle, line_width
    def render(self, draw_op, context):
        r = self.rectangle; draw_op.append(b"q n " + r.graphic_state.passthrough_per_char_instruction.encode())
        lw = r.line_width if r.line_width is not None else self.line_width
        draw_op.append(f" {lw:.6f} w {r.box.x:.6f} {r.box.y:.6f} {r.box.x2-r.box.x:.6f} {r.box.y2-r.box.y:.6f} re {'f' if r.fill_background else 'S'} Q\n".encode())

class CurveRenderUnit(RenderUnit):
    def __init__(self, curve, render_order, sub_render_order=0):
        super().__init__(render_order, sub_render_order, curve.xobj_id); self.curve = curve
    def render(self, draw_op, context):
        c = self.curve; draw_op.append(b"q n ")
        if c.relocation_transform:
            try: draw_op.append(matrix_to_bytes(tuple(float(x) for x in c.relocation_transform)))
            except: pass
        if c.ctm: draw_op.append(f" {c.ctm[0]:.6f} {c.ctm[1]:.6f} {c.ctm[2]:.6f} {c.ctm[3]:.6f} {c.ctm[4]:.6f} {c.ctm[5]:.6f} cm ".encode())
        draw_op.append(b" " + c.graphic_state.passthrough_per_char_instruction.encode() + b" ")
        pth = BitStream(b" ")
        for p in (c.pdf_original_path or c.pdf_path):
            if isinstance(p, PdfOriginalPath): p = p.pdf_path
            pth.append(f"{p.x:F} {p.y:F} {p.op} ".encode() if p.has_xy else f"{p.op} ".encode())
        draw_op.append(pth + (b" f*" if c.evenodd else b" f") + b" n Q\n")

class RenderContext:
    def __init__(self, pdf_creator, page, available_font_list, page_encoding_length_map, all_encoding_length_map, xobj_available_fonts, xobj_encoding_length_map, ctm_for_ops, check_font_exists=False):
        self.pdf_creator, self.page, self.available_font_list = pdf_creator, page, available_font_list
        self.page_encoding_length_map, self.all_encoding_length_map = page_encoding_length_map, all_encoding_length_map
        self.xobj_available_fonts, self.xobj_encoding_length_map = xobj_available_fonts, xobj_encoding_length_map
        self.ctm_for_ops, self.check_font_exists = ctm_for_ops, check_font_exists

def to_int(src): return int(re.search(r"\d+", src).group(0))
def parse_mapping(text): return [int(x.group("num"), 16) for x in re.finditer(rb"<(?P<num>[a-fA-F0-9]+)>", text)]
def apply_normalization(cmap, gid, code):
    if 0x2F00 <= code <= 0x2FD5 or 0xF900 <= code <= 0xFAFF: cmap[gid] = ord(unicodedata.normalize("NFD", chr(code)))
    else: cmap[gid] = code
def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)): yield batch
def parse_tounicode_cmap(data):
    cmap = {}
    for x in re.finditer(rb"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange", data):
        for s, t, v in batched(parse_mapping(x.group("r")), 3):
            for g in range(s, t + 1): apply_normalization(cmap, g, v + g - s)
    for x in re.finditer(rb"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", data):
        for g, c in batched(parse_mapping(x.group("c")), 2): apply_normalization(cmap, g, c)
    return cmap
def parse_truetype_data(data):
    face = freetype.Face(io.BytesIO(data)); return [i for i in range(face.num_glyphs) if face.load_glyph(i) or face.glyph.outline.contours]
def make_tounicode(cmap, used):
    line = ["/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange"]
    for b in batched([(x, cmap[x]) for x in used if x in cmap], 100):
        line.append(f"{len(b)} beginbfchar")
        for g, c in b: line.append(f"<{g:04x}><{c:04x}>" if c < 0x10000 else f"<{g:04x}><{0xD800+(c-0x10000>>10):04x}{0xDC00+(c-0x10000&0x3FF):04x}>")
        line.append("endbfchar")
    line.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]); return "\n".join(line)
def reproduce_cmap(doc):
    for pg in doc:
        for f in pg.get_fonts():
            if f[1] == "ttf" and f[3] in FONT_NAMES:
                try:
                    m, fd = doc.xref_get_key(f[0], "ToUnicode"), doc.xref_get_key(f[0], "DescendantFonts")
                    mi = to_int(m[1]); ff = doc.xref_get_key(to_int(fd[1]), "FontDescriptor/FontFile2")
                    cmap = parse_tounicode_cmap(doc.xref_stream(mi)); used = parse_truetype_data(doc.xref_stream(to_int(ff[1])))
                    doc.update_stream(mi, bytes(make_tounicode(cmap, used), "U8"))
                except: pass
    return doc

def _subset_fonts_process(p, o):
    try: pdf = pymupdf.open(p); pdf.subset_fonts(fallback=False); pdf.save(o); os._exit(0)
    except: os._exit(1)
def _save_pdf_clean_process(p, o, g=1, d=True, c=True, df=True, l=False):
    try: pymupdf.open(p).save(o, garbage=g, deflate=d, clean=c, deflate_fonts=df, linear=l); os._exit(0)
    except: os._exit(1)

class PDFCreater:
    stage_name = "Generate drawing instructions"
    def __init__(self, original_pdf_path, document, translation_config, mediabox_data):
        self.original_pdf_path, self.docs = original_pdf_path, document
        self.translation_config, self.mediabox_data = translation_config, mediabox_data
        self.font_mapper = FontMapper(translation_config)

    def render_graphic_state(self, draw_op, gs):
        if gs and gs.passthrough_per_char_instruction: draw_op.append(f"{gs.passthrough_per_char_instruction} \n".encode())

    def render_paragraph_to_char(self, paragraph: il_version_1.PdfParagraph) -> list:
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_character: chars.append(composition.pdf_character)
            elif composition.pdf_formula: chars.extend(composition.pdf_formula.pdf_character)
        
        # --- السطر الوحيد المطلوب حقنه لإصلاح الاتجاه ---
        if self.translation_config.lang_out == "ar": chars.reverse()
        
        return chars

    def create_render_units_for_page(self, page, translation_config):
        render_units = []
        chars = list(page.pdf_character or [])
        for paragraph in page.pdf_paragraph: chars.extend(self.render_paragraph_to_char(paragraph))
        for i, char in enumerate(chars): render_units.append(CharacterRenderUnit(char, getattr(char, "render_order", 100), i))
        if not translation_config.skip_form_render:
            all_forms = list(page.pdf_form or [])
            for p in page.pdf_paragraph:
                for comp in p.pdf_paragraph_composition:
                    if comp.pdf_formula: all_forms.extend(comp.pdf_formula.pdf_form)
            for i, form in enumerate(all_forms): render_units.append(FormRenderUnit(form, getattr(form, "render_order", 50), i))
        return render_units

    def render_units_to_stream(self, render_units, context, page_op, xobj_draw_ops):
        for unit in sorted(render_units, key=lambda unit: unit.get_sort_key()):
            unit.render(xobj_draw_ops.get(unit.xobj_id, page_op), context)

    def get_available_font_list(self, pdf, page): return self.get_xobj_available_fonts(pdf[page.page_number].xref, pdf)

    def get_xobj_available_fonts(self, page_xref_id, pdf):
        try:
            _, r_id = pdf.xref_get_key(page_xref_id, "Resources")
            if " 0 R" in r_id: r_id = pdf.xref_object(to_int(r_id))
            xref_id = re.search("/Font (\\d+) 0 R", r_id)
            font_dict = pdf.xref_object(int(xref_id.group(1))) if xref_id else re.search("/Font *<<(.+?)>>", r_id.replace("\n", " ")).group(1)
            return set(re.findall("/([^ ]+?) ", font_dict))
        except: return set()

    @staticmethod
    def subset_fonts_in_subprocess(pdf, config, tag):
        t_in, t_out = str(config.get_working_file_path(f"si_{tag}.pdf")), str(config.get_working_file_path(f"so_{tag}.pdf"))
        pdf.save(t_in); p = Process(target=_subset_fonts_process, args=(t_in, t_out)); p.start(); p.join(60)
        return pymupdf.open(t_out) if Path(t_out).exists() else pdf

    @staticmethod
    def save_pdf_with_timeout(pdf, output_path, translation_config, garbage=1, deflate=True, clean=True, deflate_fonts=True, linear=False, timeout=120, tag=""):
        t_in, t_out = str(translation_config.get_working_file_path(f"vi_{tag}.pdf")), str(translation_config.get_working_file_path(f"vo_{tag}.pdf"))
        pdf.save(t_in); p = Process(target=_save_pdf_clean_process, args=(t_in, t_out, garbage, deflate, clean)); p.start(); p.join(timeout)
        if Path(t_out).exists():
            import shutil; shutil.copy2(t_out, output_path); return True
        pdf.save(output_path, garbage=garbage, deflate=deflate, clean=False); return False

    def write(self, config, check_font_exists=False) -> TranslateResult:
        try:
            bn, suff = Path(config.input_file).stem, (".debug" if config.debug else "")
            m_out = config.get_output_file_path(f"{bn}{suff}.{config.lang_out}.mono.pdf")
            pdf = pymupdf.open(self.original_pdf_path); self.font_mapper.add_font(pdf, self.docs)
            with config.progress_monitor.stage_start(self.stage_name, len(self.docs.page)) as pbar:
                for page in self.docs.page: self.update_page_content_stream(check_font_exists, page, pdf, config); pbar.advance()
            if not config.skip_clean: pdf = self.subset_fonts_in_subprocess(pdf, config, "mono")
            for xref, data in self.mediabox_data.items():
                for name, box in data.items():
                    try: pdf.xref_set_key(xref, name, box)
                    except: pass
            if not config.no_mono: self.save_pdf_with_timeout(pdf, m_out, config, garbage=1, clean=not config.skip_clean, tag="mono")
            return TranslateResult(m_out, None, None)
        except:
            if not check_font_exists: return self.write(config, True)
            raise

    def update_page_content_stream(self, check_font_exists, page, pdf, translation_config, skip_char=False):
        box, avail = page.cropbox.box, self.get_available_font_list(pdf, page)
        ctm = f" 1 0 0 1 {-box.x:f} {-box.y:f} cm ".encode()
        cmap = {f.font_id: f.encoding_length for f in page.pdf_font}
        x_avail, x_ops, x_cmap = {}, {}, cmap.copy()
        for x in page.pdf_xobject:
            x_avail[x.xobj_id] = avail | self.get_xobj_available_fonts(x.xref_id, pdf)
            x_cmap[x.xobj_id] = {f.font_id: f.encoding_length for f in x.pdf_font}; x_cmap[x.xobj_id].update(cmap)
            x_ops[x.xobj_id] = BitStream(zstd_decompress(x.base_operations.value).encode())
        ctx = RenderContext(self, page, avail, cmap, x_cmap, x_avail, x_cmap, ctm, check_font_exists)
        p_op = BitStream(ctm + b" \n"); units = self.create_render_units_for_page(page, translation_config)
        if skip_char: units = [u for u in units if not isinstance(u, CharacterRenderUnit)]
        self.render_units_to_stream(units, ctx, p_op, x_ops)
        for x in page.pdf_xobject: pdf.update_stream(x.xref_id, x_ops[x.xobj_id].tobytes())
        container = pdf.get_new_xref(); pdf.update_object(container, "<<>>")
        pdf.update_stream(container, p_op.tobytes()); pdf[page.page_number].set_contents(container)