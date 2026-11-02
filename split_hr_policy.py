import json
import re
from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(__file__).with_name("HR Policy Document.pdf")
OUTPUT_PATH = Path(__file__).with_name("hr_policy_chunks.jsonl")
CHUNK_SIZE = 500
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.10)

DOCUMENT_OVERVIEW = ("General HR Policy", "Document Overview")
LEAVE_ACCRUAL = ("Leave Policy", "Leave Categories & Accrual")
GENERAL_LEAVE = ("Leave Policy", "General Leave Rules")
PUBLIC_HOLIDAYS = ("Leave Policy", "Public Holidays")
MATERNITY = ("Parental Leave Policy", "Maternity Leave")
ADOPTION = ("Parental Leave Policy", "Adoption & Surrogacy")
PATERNITY = ("Parental Leave Policy", "Paternity Leave")
HYBRID_WORK = ("Remote Work Policy", "Hybrid Work Schedule")
REMOTE_EQUIPMENT = ("Remote Work Policy", "Equipment & Home Office Allowance")
REMOTE_SECURITY = ("Remote Work Policy", "Security & Data Privacy")
INSURANCE_COVERAGE = ("Medical Insurance Policy", "Coverage & Dependents")
INSURANCE_CLAIMS = ("Medical Insurance Policy", "Enrollment & Claims")
ACKNOWLEDGMENT = ("General HR Policy", "Employee Acknowledgment")


def is_heading(line: str, heading: str) -> bool:
    return line.strip().casefold() == heading.casefold()


def metadata_for_line(line: str, current: tuple[str, str]) -> tuple[str, str]:
    heading = line.strip()
    folded = heading.casefold()

    if folded.startswith("1. employee leave policy"):
        return LEAVE_ACCRUAL
    if folded.startswith("1.1 leave categories and accrual"):
        return LEAVE_ACCRUAL
    if folded.startswith("1.2 business rules for general leave"):
        return GENERAL_LEAVE
    if folded.startswith("note on public holidays"):
        return PUBLIC_HOLIDAYS
    if folded.startswith("2. maternity & paternity leave policy"):
        return MATERNITY
    if folded.startswith("2.1 maternity leave"):
        return MATERNITY
    if is_heading(heading, "Adoption & Surrogacy"):
        return ADOPTION
    if folded.startswith("2.2 paternity leave"):
        return PATERNITY
    if folded.startswith("3. remote work & telecommuting policy"):
        return HYBRID_WORK
    if folded.startswith("3.1 hybrid working model"):
        return HYBRID_WORK
    if folded.startswith("3.2 equipment, allowance, and workspace"):
        return REMOTE_EQUIPMENT
    if folded.startswith("3.3 security and data privacy"):
        return REMOTE_SECURITY
    if folded.startswith("4. medical insurance policy"):
        return INSURANCE_COVERAGE
    if folded.startswith("4.1 coverage summary"):
        return INSURANCE_COVERAGE
    if folded.startswith("4.2 enrollment and claim process"):
        return INSURANCE_CLAIMS
    if is_heading(heading, "EMPLOYEE ACKNOWLEDGMENT"):
        return ACKNOWLEDGMENT
    return current


def extract_sections(pdf_path: Path) -> list[tuple[str, tuple[str, str]]]:
    reader = PdfReader(str(pdf_path))
    sections: list[tuple[str, tuple[str, str]]] = []
    current_metadata = DOCUMENT_OVERVIEW

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for line in page_text.splitlines():
            line = line.strip()
            if not line or re.fullmatch(r"Page \d+ of \d+", line, re.IGNORECASE):
                continue
            if re.fullmatch(r"\[Company Name\]\s+[—–-]\s+Confidential", line):
                continue

            current_metadata = metadata_for_line(line, current_metadata)
            if sections and sections[-1][1] == current_metadata:
                previous_text, _ = sections[-1]
                sections[-1] = (f"{previous_text}\n{line}", current_metadata)
            else:
                sections.append((line, current_metadata))

    return sections


def split_at_separator(text: str, separator: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    position = text.find(separator, start)
    while position != -1:
        end = position + len(separator)
        pieces.append(text[start:end])
        start = end
        position = text.find(separator, start)
    if start < len(text):
        pieces.append(text[start:])
    return pieces or [text]


def recursive_boundaries(
    text: str, separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
) -> list[int]:
    """Return preferred end positions, recursively splitting oversized text."""
    if len(text) <= CHUNK_SIZE:
        return [len(text)]

    for separator_index, separator in enumerate(separators):
        if separator and separator not in text:
            continue

        if separator:
            pieces = split_at_separator(text, separator)
        else:
            pieces = [
                text[index : index + CHUNK_SIZE]
                for index in range(0, len(text), CHUNK_SIZE)
            ]

        boundaries: list[int] = []
        offset = 0
        for piece in pieces:
            if len(piece) > CHUNK_SIZE and separator_index + 1 < len(separators):
                nested = recursive_boundaries(piece, separators[separator_index + 1 :])
                boundaries.extend(offset + boundary for boundary in nested)
            elif len(piece) > CHUNK_SIZE:
                for end in range(CHUNK_SIZE, len(piece), CHUNK_SIZE):
                    boundaries.append(offset + end)
                boundaries.append(offset + len(piece))
            else:
                boundaries.append(offset + len(piece))
            offset += len(piece)
        return sorted(set(boundaries))

    return list(range(CHUNK_SIZE, len(text), CHUNK_SIZE)) + [len(text)]


def recursively_split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    preferred_ends = recursive_boundaries(text)
    chunks: list[str] = []
    start = 0

    while start < len(text):
        limit = min(start + CHUNK_SIZE, len(text))
        eligible_ends = [end for end in preferred_ends if start < end <= limit]
        end = max(eligible_ends) if eligible_ends else limit
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def make_chunks(pdf_path: Path) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for text, (category, sub_category) in extract_sections(pdf_path):
        for chunk_text in recursively_split(text):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "category": category,
                        "sub_category": sub_category,
                    },
                }
            )
    return chunks


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Input PDF not found: {PDF_PATH}")

    chunks = make_chunks(PDF_PATH)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
