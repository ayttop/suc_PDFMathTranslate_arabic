import copy, json, logging, re, tiktoken
from pathlib import Path
from string import Template
from tqdm import tqdm

# استيرادات الدعم العربي
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from babeldoc.format.pdf.document_il import Document, Page, PdfFont, PdfParagraph
from babeldoc.format.pdf.document_il.midend import il_translator
from babeldoc.format.pdf.document_il.midend.il_translator import (
    DocumentTranslateTracker, PageTranslateTracker, ParagraphTranslateTracker, ILTranslator
)
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor

logger = logging.getLogger(__name__)

class BatchParagraph:
    def __init__(self, paragraphs, pages, page_tracker):
        self.paragraphs, self.pages, self.trackers = paragraphs, pages, [page_tracker.new_paragraph() for _ in paragraphs]

class ILTranslatorLLMOnly:
    stage_name = "Translate Paragraphs"
    def __init__(self, translate_engine, translation_config, tokenizer=None):
        self.translate_engine, self.translation_config = translate_engine, translation_config
        self.tokenizer = tokenizer or tiktoken.encoding_for_model("gpt-4o")
        self.il_translator = ILTranslator(translate_engine, translation_config, self.tokenizer)
        self.shared_context_cross_split_part = translation_config.shared_context_cross_split_part
        self.ok_count, self.fallback_count, self.total_count = 0, 0, 0

    def calc_token_count(self, text): return len(self.tokenizer.encode(text, disallowed_special=()))

    def translate(self, docs: Document) -> None:
        self.il_translator.docs = docs; tracker = DocumentTranslateTracker(); self.mid = 0
        total = sum(len([p for p in pg.pdf_paragraph if p.unicode]) for pg in docs.page)
        t_ids = set()
        with self.translation_config.progress_monitor.stage_start(self.stage_name, total) as pbar:
            with PriorityThreadPoolExecutor(max_workers=self.translation_config.pool_max_workers) as e2:
                with PriorityThreadPoolExecutor(max_workers=self.translation_config.pool_max_workers) as e:
                    for page in docs.page:
                        page_tracker = tracker.new_page()
                        fm, xm = self._build_font_maps(page)
                        batch, tokens = [], 0
                        for p in page.pdf_paragraph:
                            if id(p) in t_ids or not p.unicode: continue
                            tokens += self.calc_token_count(p.unicode); batch.append(p); t_ids.add(id(p))
                            if tokens > 200 or len(batch) > 5:
                                self.mid += 1; e.submit(self.translate_paragraph, BatchParagraph(batch, [page]*len(batch), page_tracker), pbar, fm, xm, None, None, e2, 0, tokens, self.mid)
                                batch, tokens = [], 0
                        if batch:
                            self.mid += 1; e.submit(self.translate_paragraph, BatchParagraph(batch, [page]*len(batch), page_tracker), pbar, fm, xm, None, None, e2, 0, tokens, self.mid)

    def _build_font_maps(self, pg):
        pf = {f.font_id: f for f in pg.pdf_font}
        xf = {x.xobj_id: {**pf, **{f.font_id: f for f in x.pdf_font}} for x in pg.pdf_xobject}
        return pf, xf

    def translate_paragraph(self, batch_paragraph, pbar, page_font_map, xobj_font_map, title_paragraph, local_title_paragraph, executor, priority, paragraph_token_count, mp_id, *args, **kwargs):
        self.translation_config.raise_if_cancelled(); inputs = []
        try:
            for i, p in enumerate(batch_paragraph.paragraphs):
                txt, ti = self.il_translator.pre_translate_paragraph(p, batch_paragraph.trackers[i], page_font_map, xobj_font_map)
                if txt: inputs.append({"txt": txt, "ti": ti, "p": p, "tr": batch_paragraph.trackers[i]})
                else: pbar.advance(1)
            if not inputs: return
            json_in = json.dumps([{"id": i, "input": inp["txt"]} for i, inp in enumerate(inputs)], ensure_ascii=False)
            res = self.translate_engine.llm_translate(f"Translate to Arabic JSON: {json_in}", {"paragraph_token_count": paragraph_token_count, "request_json_mode": True}).strip()
            parsed = json.loads(res.strip().strip("```json").strip("```").strip())
            for it in (parsed if isinstance(parsed, list) else [parsed]):
                idx, out = int(it["id"]), it.get("output", it.get("input"))
                
                # --- حقن إصلاح العربية هنا ---
                if self.translation_config.lang_out == "ar":
                    try: out = get_display(reshape(out))
                    except: pass
                
                self.il_translator.post_translate_paragraph(inputs[idx]["p"], inputs[idx]["tr"], inputs[idx]["ti"], out)
                self.ok_count += 1; pbar.advance(1)
            self.total_count += len(inputs)
        except Exception:
            for inp in inputs:
                try: executor.submit(self.il_translator.translate_paragraph, inp["p"], batch_paragraph.pages[0], pbar, inp["tr"], page_font_map, xobj_font_map, 0, 0)
                except: pass