from pathlib import Path
# a powerful XML parser — reads .xml files and lets us navigate their tags.
from lxml import etree
# helps us generate unique hashes for each text chunk (for caching and versioning later).
import hashlib
import json


MAX_CHARS = 1200

IGNORABLE = {"legal", "copyright"}


ns = {
    'db': 'http://docbook.org/ns/docbook',
}


def text_of(node):
    # Convert XML node to readable text (keeps simple bullets/numbers)
    # node.itertext() walks through all text inside the XML node (including nested tags).
    text = ''.join(''.join(node.itertext()).split())
    return text.strip()


def chunk_text(text, max_chars=MAX_CHARS):
    if len(text) <= max_chars:
        return [text]

# parts → will hold the final chunks (each piece of text after splitting) cur → will hold the current sentences we’re building up before turning them into a chunk
    parts, cur = [], []
    for sent in text.split('.'):
        # → computes the total number of characters already in the current chunk (cur list). We add +2 for each sentence to roughly count the extra period and space we’ll put between sentences when we join them later.
        if sum(len(s)+2 for s in cur)+len(sent) < max_chars:
            cur.append(sent)
        else:
            parts.append('.'.join(cur))
            cur = [sent]

        # Without this check, you’d lose the last chunk of text if it never reached the size limit.
    if cur:
        parts.append('.'.join(cur))
    return parts


_counter = 0


def make_chunk(doc_id, path, text, meta, xml_file):
    global _counter
    _counter += 1
    cid = f"{doc_id}::{_counter}"
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    return {
        'id': cid,
        'doc_id': doc_id,
        'path': path,
        'title': path.split('>')[-1].strip(),
        'text': text,
        'lang': meta.get('lang', 'en'),
        'product': meta.get('product', 'generic'),
        'version': meta.get('version', 'v1'),
        'audience': meta.get('audience', ''),
        'tags': meta.get('tags', []),
        'conditions_json': json.dumps(meta.get('conditions', {}), ensure_ascii=False),
        'rev': meta.get('rev', ''),
        'hash': h,
        'source_file': str(xml_file)
    }


def extract_chunks(xml_dir: Path, meta_defaults: dict):
    chunks = []
    for xml_file in xml_dir.rglob('*.xml'):
        root = etree.parse(str(xml_file)).getroot()
        topic_title = root.find('.//db:title', namespaces=ns)
        topic_title = text_of(
            topic_title) if topic_title is not None else xml_file.stem
        doc_id = root.get('id') or xml_file.stem
        sections = root.findall('.//db:section', namespaces=ns)
        if not sections:
            sections = [root]  # treat whole doc as one section
            # Get the first paragraph (intro).Combine title + paragraph → one piece of meaningful context.If it’s long, chunk_text will split it.Add each chunk to our chunks list.
            lead_para = root.find('.//db:para', namespaces=ns)
            if lead_para is not None:
                lead = f"{topic_title}: {text_of(lead_para)}"
                for t in chunk_text(lead):
                    chunks.append(make_chunk(
                        doc_id, topic_title, t, meta_defaults, xml_file))

        for sec in sections:
            sec_title_el = sec.find('.//db:title', namespaces=ns)
            sec_title = text_of(
                sec_title_el) if sec_title_el is not None else topic_title
            body_text = text_of(sec)
            if not body_text:
                continue
            if sec.get('role') in IGNORABLE:
                continue
        for t in chunk_text(f"{topic_title} > {sec_title} : {body_text}"):
            chunks.append(make_chunk(
                doc_id, f"{topic_title} > {sec_title}", t, meta_defaults, xml_file)),
    return chunks


_counter = 0
