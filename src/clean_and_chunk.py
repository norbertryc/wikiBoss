# code mostly AI-generated

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import html
import math
import hashlib
import mwparserfromhell

# --- Settings and patterns ---
CATEGORY_PREFIXES = ("Kategoria:", "Category:")
FILE_PREFIXES = ("Plik:", "File:", "Grafika:", "Image:", "Obraz:")
REF_AND_COMMENTS_RE = re.compile(r"(?is)<!--.*?-->|<ref\b[^/>]*?/>|<ref\b[^>]*?>.*?</ref>")
HTML_TAGS_RE = re.compile(r"(?is)<[^>]+>")
ATTR_PAIR_RE = re.compile(
    r'\s*(?:class|style|align|bgcolor|width|height|border|cellpadding|cellspacing|colspan|rowspan)'
    r'\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^|!\n]+)'
)
CAT_LINE_RE = re.compile(r"\[\[\s*(Kategoria|Category)\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)

# Helper sections PL -> meta keys
HELPER_SECTIONS = {
    "zobacz też": "see_also",
    "przypisy": "references_section",
    "uwagi": "notes",
    "bibliografia": "bibliography",
    "linki zewnętrzne": "external_links_section",
}

# --- HTML entities / non-breaking spaces ---
def unescape_html_spaces(text: str) -> str:
    t = html.unescape(text)
    return t.replace("\u00a0", " ")

# --- Tables: placeholders and cleanup ---
TABLE_PH = "§§TABLE_{}§§"
PURE_TABLE_RE = re.compile(r"^\s*\{\|[\s\S]*?\|\}\s*$")
CODE_PH = "§§CODEBLOCK_{}§§"
INLINE_CODE_TAGS = {"code", "tt", "kbd", "samp", "var", "nowiki"}
BLOCK_CODE_TAGS  = {"pre", "syntaxhighlight", "source"}

def reinsert_codeblocks(text: str, blocks: list[str]) -> str:
    # restore code exactly as it was (keep indentation and spaces)
    for i, block in enumerate(blocks):
        # wrap with a new line to avoid gluing with adjacent text
        text = text.replace(CODE_PH.format(i), "\n" + block + "\n")
    return text

def extract_tables_with_placeholders(wikitext: str) -> Tuple[str, List[str]]:
    out = []
    i = 0
    depth = 0
    start = None
    tables = []
    while i < len(wikitext):
        if wikitext.startswith("{|", i):
            if depth == 0:
                start = i
            depth += 1
            i += 2
            continue
        if depth > 0 and wikitext.startswith("|}", i):
            depth -= 1
            i += 2
            if depth == 0 and start is not None:
                tbl = wikitext[start:i]
                placeholder = TABLE_PH.format(len(tables))
                tables.append(tbl)
                out.append(placeholder)
                start = None
            continue
        if depth == 0:
            out.append(wikitext[i])
        i += 1
    return "".join(out), tables

def clean_wikitable_attributes(tbl_text: str) -> str:
    lines = tbl_text.splitlines()
    cleaned = []
    for line in lines:
        line = HTML_TAGS_RE.sub("", line)
        line = unescape_html_spaces(line)
        ls = line.lstrip()
        if ls.startswith("{|"):
            cleaned.append("{|")
            continue
        if ls.startswith("|-"):
            cleaned.append("|-")
            continue
        if ls.startswith("|}"):
            cleaned.append("|}")
            continue
        line2 = ATTR_PAIR_RE.sub("", line)
        line2 = re.sub(r"\s+\|\s+", " | ", line2)
        line2 = re.sub(r"\s+\|\|\s+", " || ", line2)
        cleaned.append(line2.rstrip())
    text = "\n".join([ln for ln in cleaned if ln.strip() != ""])
    return text

def reinsert_tables_with_cleanup(text: str, tables: List[str]) -> str:
    for i, tbl in enumerate(tables):
        cleaned_tbl = clean_wikitable_attributes(tbl)
        text = text.replace(TABLE_PH.format(i), "\n" + cleaned_tbl + "\n")
    return text

def is_pure_table(text: str) -> bool:
    if not PURE_TABLE_RE.match(text):
        return False
    return text.count("{|") == 1 and text.count("|}") == 1

# --- Helpers: text and headings ---
def normalize_visible_text(s: str) -> str:
    s = unescape_html_spaces(s)
    # do not replace all space sequences – only tabs to a space and excessive multiple newlines
    s = s.replace("\t", " ")
    s = re.sub(r"\n{3,}", "\n\n", s)
    # remove only truly empty parentheses
    s = re.sub(r"\(\s*\)", "", s)
    return s.strip()

def get_heading_and_level(section_code: mwparserfromhell.wikicode.Wikicode) -> Tuple[str, int]:
    heads = section_code.filter_headings(recursive=False)
    if heads:
        h = heads[0]
        title = str(h.title.strip_code()).strip()
        level = int(getattr(h, "level", 2) or 2)
        return title, level
    return "", 0  # lead

def normalize_heading_pl(h: str) -> str:
    return (h or "").strip().lower()

# --- Expanding selected templates (K/J) ---
def expand_known_templates_inplace(code: mwparserfromhell.wikicode.Wikicode) -> None:
    for tpl in list(code.filter_templates(recursive=True)):
        name = str(tpl.name.strip_code()).strip()
        if name in {"K", "J"}:
            pos_vals = []
            for p in tpl.params:
                if p.showkey:
                    continue
                val = str(p.value.strip_code()).strip()
                if val.lower() in {"pl", "en", "de", "fr", "es"}:
                    continue
                if val:
                    pos_vals.append(val)
            display = pos_vals[-1] if pos_vals else ""
            if display:
                try:
                    code.replace(tpl, mwparserfromhell.parse(display), recursive=True)
                except Exception:
                    pass

def remove_images_and_collect_links(code: mwparserfromhell.wikicode.Wikicode) -> Dict[str, Any]:
    meta = {"external_links": []}
    for el in list(code.filter_external_links(recursive=True)):
        url = str(getattr(el, "url", el))
        meta["external_links"].append(url)
        try:
            code.remove(el, recursive=True)
        except Exception:
            pass
    for wl in list(code.filter_wikilinks(recursive=True)):
        target = str(wl.title.strip_code()).strip()
        if any(target.lower().startswith(pfx.lower()) for pfx in FILE_PREFIXES):
            try:
                code.remove(wl, recursive=True)
            except Exception:
                pass
    return meta

# --- Section content cleanup (preserving tables) ---
HATNOTE_TEMPLATES = {"inne znaczenia", "uwaga", "disambig", "other uses", "see also"}

def clean_section_to_text(section_code: mwparserfromhell.wikicode.Wikicode) -> Tuple[str, Dict[str, Any]]:
    meta = {"wikilinks": [], "external_links": [], "categories": []}

    raw = str(section_code)
    raw_wo_tables, tables = extract_tables_with_placeholders(raw)
    raw_sanitized = REF_AND_COMMENTS_RE.sub("", raw_wo_tables)

    code = mwparserfromhell.parse(raw_sanitized, skip_style_tags=True)

    expand_known_templates_inplace(code)
    link_meta = remove_images_and_collect_links(code)
    meta["external_links"] = link_meta["external_links"]

    # categories and wikilinks (unchanged)
    for wl in list(code.filter_wikilinks(recursive=True)):
        target = str(wl.title.strip_code()).strip()
        if any(target.lower().startswith(p.lower()) for p in CATEGORY_PREFIXES):
            meta["categories"].append(target)
            try:
                code.remove(wl, recursive=True)
            except Exception:
                pass
        else:
            label = str((wl.text or wl.title).strip_code()).strip()
            meta["wikilinks"].append({"target": target, "text": label if label != target else None})

    # hatnotes/templates to remove (unchanged)
    for tpl in list(code.filter_templates(recursive=True)):
        name = str(tpl.name).strip().lower()
        if name in HATNOTE_TEMPLATES:
            try:
                code.remove(tpl, recursive=True)
            except Exception:
                pass

    # remove headings from body (unchanged)
    for hd in list(code.filter_headings(recursive=True)):
        try:
            code.remove(hd, recursive=True)
        except Exception:
            pass

    # --- NEW: protect inline and block code ---
    codeblocks: list[str] = []
    for tag in list(code.filter_tags(recursive=True)):
        tname = str(tag.tag).strip().lower()

        # ref already removed on raw, but just in case
        if tname == "ref":
            code.replace(tag, "", recursive=True)
            continue

        # <br> -> newline
        if tname == "br":
            code.replace(tag, mwparserfromhell.parse("\n"), recursive=True)
            continue

        # inline code: replace with contents
        if tname in INLINE_CODE_TAGS:
            code.replace(tag, tag.contents if tag.contents is not None else "", recursive=True)
            continue

        # block code: store and insert placeholder
        if tname in BLOCK_CODE_TAGS:
            block_raw = str(tag.contents) if tag.contents is not None else ""
            block_raw = unescape_html_spaces(block_raw)
            codeblocks.append(block_raw)
            ph = CODE_PH.format(len(codeblocks) - 1)
            code.replace(tag, mwparserfromhell.parse(ph), recursive=True)
            continue

        # other tags: keep only content (do not drop letters!)
        code.replace(tag, tag.contents if tag.contents is not None else "", recursive=True)

    # Note: disable collapse to avoid merging spaces; placeholders will protect code blocks
    clean_text = code.strip_code(normalize=True, collapse=False, keep_template_params=False)
    clean_text = normalize_visible_text(clean_text)

    # restore code blocks (after normalization)
    clean_text = reinsert_codeblocks(clean_text, codeblocks)

    # restore tables after cleanup
    clean_text = reinsert_tables_with_cleanup(clean_text, tables)

    return clean_text, meta

# --- Splitting long fragments (without cutting tables) ---
_SENT_SPLIT_RE = re.compile(r"(?<=\S[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])")

def split_text_balanced(text: str,
                        target_len: int = 1500,
                        max_len: int = 3000,
                        min_tail: int = 600) -> List[str]:
    if is_pure_table(text):
        return [text]
    if len(text) <= max_len:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    n = max(2, math.ceil(len(text) / target_len))
    chunk_size = min(max_len, math.ceil(len(text) / n))
    parts: List[str] = []
    buf: List[str] = []
    acc = 0
    remaining_total = len(text)

    def flush_buf():
        nonlocal buf, acc
        if buf:
            parts.append("\n\n".join(buf).strip())
            buf = []
            acc = 0

    for para in paragraphs:
        if is_pure_table(para):
            flush_buf()
            parts.append(para)
            remaining_total -= len(para) + 2
            continue
        para_len = len(para) + (2 if buf else 0)
        remaining_after = remaining_total - para_len
        if acc >= chunk_size and remaining_after >= min_tail:
            flush_buf()
        if para_len > max_len:
            sentences = _SENT_SPLIT_RE.split(para)
            cur = ""
            for s in sentences:
                add = ("" if not cur else " ") + s.strip()
                if len(cur) + len(add) > max_len:
                    if cur:
                        if buf:
                            buf.append(cur)
                        else:
                            buf = [cur]
                        flush_buf()
                        cur = s.strip()
                    else:
                        parts.append(s.strip())
                        cur = ""
                else:
                    cur += add
            if cur:
                if buf:
                    buf.append(cur)
                else:
                    buf = [cur]
            acc = sum(len(x) for x in buf) + max(0, 2 * (len(buf) - 1))
            remaining_total = remaining_after
            continue
        buf.append(para)
        acc += para_len
        remaining_total = remaining_after
        if acc >= chunk_size and remaining_total >= min_tail:
            flush_buf()
    if buf:
        tail = "\n\n".join(buf).strip()
        if parts and len(tail) < min_tail and len(parts[-1]) + 2 + len(tail) <= max_len + 400:
            parts[-1] = (parts[-1] + "\n\n" + tail).strip()
        else:
            parts.append(tail)
    return parts

# --- Article meta: categories, references from tags, helper sections PL ---
def extract_categories_from_tail(wikitext: str) -> List[str]:
    cats = []
    for m in CAT_LINE_RE.finditer(wikitext):
        raw = m.group(2).strip()
        name = raw.split("|", 1)[0].strip()  # remove sort key / trailing pipe
        if name:
            cats.append(name)
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _attr_value_to_text(aval) -> str:
    if aval is None:
        return ""
    try:
        return str(aval.strip_code(normalize=True, collapse=True)).strip()
    except Exception:
        return str(aval).strip()

def collect_reference_names(wikitext: str) -> List[str]:
    code = mwparserfromhell.parse(wikitext, skip_style_tags=True)
    names: List[str] = []
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() != "ref":
            continue
        name_val = ""
        attrs = getattr(tag, "attributes", None)
        if attrs:
            # attributes may be a list of objects, iterate defensively
            for attr in attrs:
                try:
                    aname = str(getattr(attr, "name", "")).strip().lower()
                    if aname == "name":
                        name_val = _attr_value_to_text(getattr(attr, "value", ""))
                        break
                except Exception:
                    continue
        if name_val:
            name_val = unescape_html_spaces(name_val)  # remove e.g. &nbsp;
            if name_val not in names:
                names.append(name_val)
    return names

def collect_helper_section_content(section_code: mwparserfromhell.wikicode.Wikicode,
                                   heading_norm: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    code = mwparserfromhell.parse(str(section_code), skip_style_tags=True)
    # remove heading
    for hd in list(code.filter_headings(recursive=True)):
        try:
            code.remove(hd, recursive=True)
        except Exception:
            pass

    if heading_norm == "zobacz też":
        items = []
        for wl in code.filter_wikilinks(recursive=True):
            target = str(wl.title.strip_code()).strip()
            label = str((wl.text or wl.title).strip_code()).strip()
            # filter out categories/images
            if target.lower().startswith(("kategoria:", "category:")):
                continue
            if any(target.lower().startswith(pfx.lower()) for pfx in FILE_PREFIXES):
                continue
            items.append({"target": target, "text": label if label != target else None})
        data["see_also"] = items
    elif heading_norm == "linki zewnętrzne":
        links = []
        for el in code.filter_external_links(recursive=True):
            links.append({
                "url": str(getattr(el, "url", "") or ""),
                "title": str(el.title.strip_code()).strip() if getattr(el, "title", None) else ""
            })
        data["external_links"] = links
    elif heading_norm == "bibliografia":
        # Extract items as list lines (asterisks) after strip_code
        raw = code.strip_code(normalize=True, collapse=True)
        raw = normalize_visible_text(raw)
        items = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith(("*", "•", "–", "-")):
                s = s.lstrip("*•–- ").strip()
            items.append(s)
        data["bibliography"] = items
    elif heading_norm == "uwagi":
        raw = code.strip_code(normalize=True, collapse=True)
        raw = normalize_visible_text(raw)
        notes = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            notes.append(s)
        data["notes"] = notes
    elif heading_norm == "przypisy":
        # Presence of the section + any text (rare) – references are collected from <ref> anyway
        raw = code.strip_code(normalize=True, collapse=True)
        raw = normalize_visible_text(raw)
        data["has_references_section"] = True
        if raw:
            data["references_section_text"] = raw
    return data

# --- Sectioning + split, moving helper sections to meta ---
def _make_section_uid(title: str, idx: int) -> str:
    base = f"{idx}::{title or 'LEAD'}".encode("utf-8")
    return hashlib.md5(base).hexdigest()[:10]

def chunk_and_collect_meta(wikitext: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    categories = extract_categories_from_tail(wikitext)
    references = collect_reference_names(wikitext)
    article_meta: Dict[str, Any] = {
        "category": categories,
        "references": references,
        "see_also": [],
        "notes": [],
        "bibliography": [],
        "external_links": [],
        "has_references_section": False,
    }

    code = mwparserfromhell.parse(wikitext, skip_style_tags=True)
    sections = code.get_sections(include_lead=True, include_headings=True)
    chunks: List[Dict[str, Any]] = []

    last_h2_title: str | None = None
    last_h2_uid: str | None = None

    for idx, sec in enumerate(sections):
        heading, level = get_heading_and_level(sec)
        hnorm = normalize_heading_pl(heading)

        # update parent context when entering a new H2
        if level == 2:
            last_h2_title = heading or ""
            last_h2_uid = _make_section_uid(last_h2_title, idx)

        # helper sections -> to meta
        if hnorm in HELPER_SECTIONS:
            data = collect_helper_section_content(sec, hnorm)
            if "see_also" in data:
                seen = {(x["target"], x.get("text")) for x in article_meta["see_also"]}
                for x in data["see_also"]:
                    key = (x["target"], x.get("text"))
                    if key not in seen:
                        article_meta["see_also"].append(x); seen.add(key)
            if "external_links" in data:
                seen = {(x["url"], x.get("title")) for x in article_meta["external_links"]}
                for x in data["external_links"]:
                    key = (x["url"], x.get("title"))
                    if key not in seen:
                        article_meta["external_links"].append(x); seen.add(key)
            if "bibliography" in data:
                for it in data["bibliography"]:
                    if it and it not in article_meta["bibliography"]:
                        article_meta["bibliography"].append(it)
            if "notes" in data:
                for it in data["notes"]:
                    if it and it not in article_meta["notes"]:
                        article_meta["notes"].append(it)
            if data.get("has_references_section"):
                article_meta["has_references_section"] = True
            if data.get("references_section_text"):
                article_meta["references_section_text"] = data["references_section_text"]
            continue

        # Cleaned section text
        text, meta = clean_section_to_text(sec)
        if level == 2:
            # compute whether this H2 contains any H3 subsections
            h3_children = sec.get_sections(levels=[3], include_headings=True, flat=True)
            has_h3_children = len(h3_children) > 0

            # if H2 has only H3 (i.e., no content after cleanup), skip H2 and emit only H3
            if not text:
                # H3 children
                for i3, sec3 in enumerate(h3_children):
                    h3_title, h3_level = get_heading_and_level(sec3)
                    h3_text, h3_meta = clean_section_to_text(sec3)
                    if not h3_text:
                        continue
                    parts3 = split_text_balanced(h3_text, target_len=1500, max_len=3000, min_tail=600)
                    h3_uid = _make_section_uid(f"{last_h2_uid}/{h3_title}", i3)
                    for p_idx, part in enumerate(parts3, start=1):
                        chunks.append({
                            "section_index": idx,
                            "section_uid": h3_uid,
                            "heading": h3_title,
                            "level": h3_level,
                            "text": part,
                            "meta": {
                                "wikilinks": h3_meta["wikilinks"],
                                "external_links": h3_meta["external_links"],
                                "char_len": len(part),
                                "link_counts": {
                                    "wikilinks": len(h3_meta["wikilinks"]),
                                    "external": len(h3_meta["external_links"]),
                                }
                            },
                            "part_index": p_idx,
                            "part_total": len(parts3),
                            "parent_h2_title": last_h2_title,
                            "parent_h2_uid": last_h2_uid,
                        })
                continue  # do not emit H2 chunk

            # H2 has content – emit as before, with has_h3_children flag
            section_uid = _make_section_uid(heading, idx)
            parts2 = split_text_balanced(text, target_len=1500, max_len=3000, min_tail=600)
            for p_idx, part in enumerate(parts2, start=1):
                chunks.append({
                    "section_index": idx,
                    "section_uid": section_uid,
                    "heading": heading,
                    "level": level,
                    "text": part,
                    "meta": {
                        "wikilinks": meta["wikilinks"],
                        "external_links": meta["external_links"],
                        "char_len": len(part),
                        "link_counts": {
                            "wikilinks": len(meta["wikilinks"]),
                            "external": len(meta["external_links"]),
                        }
                    },
                    "part_index": p_idx,
                    "part_total": len(parts2),
                    "parent_h2_title": None,
                    "parent_h2_uid": None,
                    "has_h3_children": has_h3_children,
                })
            continue

        # Non-H2 sections (including H3) – emit normally, but add parent if H3
        if not text:
            continue
        section_uid = _make_section_uid(heading, idx)
        parts = split_text_balanced(text, target_len=1500, max_len=3000, min_tail=600)
        parent_title = last_h2_title if level == 3 else None
        parent_uid = last_h2_uid if level == 3 else None
        for p_idx, part in enumerate(parts, start=1):
            chunks.append({
                "section_index": idx,
                "section_uid": section_uid,
                "heading": heading,
                "level": level,
                "text": part,
                "meta": {
                    "wikilinks": meta["wikilinks"],
                    "external_links": meta["external_links"],
                    "char_len": len(part),
                    "link_counts": {
                        "wikilinks": len(meta["wikilinks"]),
                        "external": len(meta["external_links"]),
                    }
                },
                "part_index": p_idx,
                "part_total": len(parts),
                "parent_h2_title": parent_title,
                "parent_h2_uid": parent_uid,
            })

    return chunks, article_meta

# --- JSONL processing ---
def process_data(in_path: str, out_path: str, id_max: int | None = None) -> None:
    in_p = Path(in_path)
    out_p = Path(out_path)
    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if id_max is not None and isinstance(rec.get("id"), int) and rec["id"] >= id_max:
                continue
            text = rec.get("text", "") or ""
            chunks, article_meta = chunk_and_collect_meta(text)
            out = {
                "id": rec.get("id"),
                "title": rec.get("title"),
                "url": rec.get("url"),
                "category": article_meta.pop("category"),  # according to your key “category”
                "num_chunks": len(chunks),
                "chunks": chunks,
                "article_meta": article_meta,  # see_also, notes, bibliography, external_links, references, etc.
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")