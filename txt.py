from pathlib import Path

import requests
from bs4 import BeautifulSoup


WIKIPEDIA_TARGETS = [
    ("https://en.wikipedia.org/wiki/Google", Path("docs/google.txt")),
    ("https://en.wikipedia.org/wiki/Microsoft", Path("docs/microsoft.txt")),
    ("https://en.wikipedia.org/wiki/Nvidia", Path("docs/nvidia.txt")),
    ("https://en.wikipedia.org/wiki/Tesla,_Inc.", Path("docs/tesla.txt")),
    ("https://en.wikipedia.org/wiki/SpaceX", Path("docs/spacex.txt")),
]

SKIP_SECTIONS = {"See also", "References", "External links", "Notes", "Further reading"}


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def fetch_wikipedia_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python requests scraper"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content_candidates = soup.select("div.mw-parser-output")
    if not content_candidates:
        raise ValueError("Could not find Wikipedia main content on the page.")
    content = max(content_candidates, key=lambda tag: len(tag.find_all("p")))

    # Remove citations, tables, and other non-article blocks before parsing sections.
    for element in content.select(
        "sup.reference, table, style, script, .hatnote, .reflist, .navbox, .toc"
    ):
        element.decompose()

    sections = []
    current_title = "Introduction"
    current_paragraphs = []

    for child in content.find_all(["h2", "h3", "p"]):
        if child.name in {"h2", "h3"}:
            if current_paragraphs:
                sections.append((current_title, current_paragraphs))
            heading = child.get_text(" ", strip=True).replace("[edit]", "").strip()
            if heading in SKIP_SECTIONS:
                break
            current_title = heading
            current_paragraphs = []
            continue

        paragraph = normalize_text(child.get_text(" ", strip=True))
        if paragraph:
            current_paragraphs.append(paragraph)

    if current_paragraphs:
        sections.append((current_title, current_paragraphs))

    if not sections:
        raise ValueError("Wikipedia page was fetched, but no structured text was extracted.")

    formatted_sections = []
    for title, paragraphs in sections:
        if not paragraphs:
            continue
        section_text = f"{title}\n\n" + "\n\n".join(paragraphs)
        formatted_sections.append(section_text)

    return "\n\n".join(formatted_sections)


def main() -> None:
    for url, output_file in WIKIPEDIA_TARGETS:
        try:
            text = fetch_wikipedia_text(url)
            output_file.write_text(text, encoding="utf-8")
            print(f"Saved successfully to {output_file.resolve()}")
        except requests.RequestException as exc:
            print(f"Download failed for {url}: {exc}")
        except Exception as exc:
            print(f"Script failed for {url}: {exc}")


if __name__ == "__main__":
    main()
