# tools/check_xml_data.py
from pathlib import Path
from lxml import etree
import pandas as pd

XML_DIR = Path("data/xml_export")

# Common namespaces you may see
NS = {
    "db": "http://docbook.org/ns/docbook",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


def first_text(root, xpaths: list[str]) -> str:
    """
    Try multiple XPath expressions in order, return cleaned text of the first hit.
    Returns '' if nothing found.
    """
    for xp in xpaths:
        nodes = root.xpath(xp, namespaces=NS)
        if not nodes:
            continue
        # node may be Element or string
        if isinstance(nodes[0], str):
            txt = nodes[0]
        else:
            txt = " ".join(nodes[0].itertext())
        txt = " ".join(txt.split()).strip()
        if txt:
            return txt
    return ""


def inspect_xml_file(path: Path):
    try:
        root = etree.parse(str(path)).getroot()
    except Exception as e:
        return {"file": path.name, "error": str(e)}

    # Helpful for debugging:
    # print(path.name, root.tag, root.nsmap)

    # Title: try namespaced DocBook, then non-ns, then local-name fallback
    topic_title = first_text(root, [
        ".//db:title",                # DocBook namespaced
        ".//title",                   # no namespace
        ".//*[local-name()='title']",  # any prefix/default ns
    ]) or path.stem.replace("_", " ").replace("-", " ").title()

    # Language: try element, then xml:lang attribute, then empty
    lang = first_text(root, [
        ".//db:language",
        ".//language",
        ".//*[local-name()='language']",
    ])
    if not lang:
        # xml:lang attribute on the root or topic element
        lang = root.get(f"{{{NS['xml']}}}lang", "")  # e.g., 'en', 'sv'

    # Product & version: try common spots; if missing, return ''
    product = first_text(root, [
        ".//db:info/db:product", ".//product", ".//*[local-name()='product']",
    ])
    version = first_text(root, [
        ".//db:info/db:version", ".//version", ".//*[local-name()='version']",
    ])

    # Count sections using ns-agnostic local-name()
    num_sections = len(root.xpath(".//*[local-name()='section']"))

    # Sample 1–3 sections (ns-agnostic)
    samples = []
    for sec in root.xpath(".//*[local-name()='section']")[:3]:
        sec_title = first_text(sec, [
            "./db:title", "./title", "./*[local-name()='title']",
        ])
        paras = sec.xpath(".//*[local-name()='para']")
        first_para = ""
        if paras:
            first_para = " ".join(paras[0].itertext()).strip()
        samples.append((sec_title, first_para))

    return {
        "file": path.name,
        "topic_title": topic_title or "",   # avoid None → NaN
        "lang": lang or "",
        "product": product or "",
        "version": version or "",
        "num_sections": num_sections,
        "sample": samples,
    }


def main():
    files = sorted(XML_DIR.glob("*.xml"))
    rows = [inspect_xml_file(f) for f in files]

    df = pd.DataFrame(rows)
    # Ensure no NaN in these columns
    for col in ["topic_title", "lang", "product", "version"]:
        if col in df:
            df[col] = df[col].fillna("")

    pd.set_option("display.max_colwidth", 120)
    print("\n--- Summary ---")
    cols = [c for c in ["file", "topic_title", "lang", "product",
                        "version", "num_sections", "error"] if c in df.columns]
    print(df[cols])

    # Show example sections for the first OK file
    print("\n--- Example Sections ---")
    for r in rows:
        if "error" not in r:
            for (t, p) in r["sample"]:
                print(f"Section: {t or '(no title)'}")
                print(" →", (p or "")[:200])
                print()
            break


if __name__ == "__main__":
    main()
