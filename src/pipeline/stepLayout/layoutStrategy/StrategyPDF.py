import pdfplumber
from collections import defaultdict, Counter
from .AbstractStrategyLayout import AbstractStrategyLayout
from ..postprocessor.LayoutPostprocessor import rows_are_similar, create_bounding_box


class StrategyPDF(AbstractStrategyLayout):
    def __init__(self, pdf_path, words=None, log=False):
        super().__init__(image=None, log=log)
        self.pdf_path = pdf_path
        self.words = words  # reuse words from earlier step

    def execute(self):
        if self.words is not None:
            words = self.words
        else:
            with pdfplumber.open(self.pdf_path) as pdf:
                page = pdf.pages[0]
                words = page.extract_words(extra_attrs=["size", "fontname"])

        # group individual OCR words into phrases
        grouped_phrases = group_words(words)
        #  group phrases into  rows
        row_structures = group_into_rows(grouped_phrases)
        # detect tables based on row similaryand alignment
        logical_tables = split_rows_into_logical_tables(row_structures)
        # detect title and section headers based on size and font
        title, section_headers = find_titles_and_headers(grouped_phrases, logical_tables)

        # convert detected tables to JSON
        tables_json = []
        for idx, table in enumerate(logical_tables):
            # flatten all cells to compute bounding box
            flat_cells = [cell for row in table for cell in row]
            bbox = create_bounding_box(flat_cells)
            table_json = {
                "bbox": bbox,
                "rows": []
            }
            for row in table:
                row_json = []
                for cell in row:
                    row_json.append({
                        "text": cell["text"],
                        "bbox": cell["bbox"]
                    })
                table_json["rows"].append(row_json)
            tables_json.append(table_json)

        # mark used text elements
        used_texts = set()
        if title:
            used_texts.add(title["text"])
        for header in section_headers:
            used_texts.add(header["text"])
        for table in logical_tables:
            for row in table:
                for cell in row:
                    used_texts.add(cell['text'])

        # find unmatched phrases (not in title,,header,table)
        unmatched_phrases = []
        for phrase in grouped_phrases:
            if all(t['text'] not in used_texts for t in phrase['tokens']):
                unmatched_phrases.append({
                    "rows": phrase['text'],
                    "bbox": phrase['bbox']
                })

        # structure with all 11  layout keys
        json_output = {
            "Table": tables_json,
            "Caption": [],
            "Footnote": [],
            "Formula": [],
            "List-item": [],
            "Page-footer": [],
            "Page-header": [],
            "Section-header": [
                {
                    "rows": [[{
                        "text": header["text"],
                        "bbox": header["bbox"]
                    }]],
                    "bbox": header["bbox"]
                } for header in section_headers
            ],
            "Text": [],
            "Title": [{
                "rows": [[{
                    "text": title["text"],
                    "bbox": title["bbox"]
                }]],
                "bbox": title["bbox"]
            }] if title else [],
            "unmatched": unmatched_phrases
        }
        return None, json_output  # no image

def find_titles_and_headers(phrases, logical_tables, bold_keywords=("Bold", "Black", "Heavy", "Demi", "Semi", "Medium")):
    # all font sizes
    sizes = [round(t.get("tokens")[0].get("size", 0), 1) for t in phrases if
             t.get("tokens") and "size" in t["tokens"][0]]
    if not sizes:
        return None, []

    # find common font size and maximum font size
    size_counts = Counter(sizes)
    dominant_size = size_counts.most_common(1)[0][0]

    max_size = max(sizes)
    min_title_size = dominant_size + 3  #  threshold for title heuristic

    title_candidate = None
    section_headers = []

    # text strings that are in a table
    table_elements = set()
    for table in logical_tables:
        for row in table:
            for cell in row:
                table_elements.add(cell['text'])

    for phrase in phrases:
        tokens = phrase.get("tokens", [])
        if not tokens:
            continue

        size = tokens[0].get("size")
        fontname = tokens[0].get("fontname", "")

        if size is None:
            continue

        phrase['size'] = size
        phrase['fontname'] = fontname

        # Skip anything that is in a table
        if phrase['text'] in table_elements:
            continue

        # If font size is large enough -> title
        if size >= min_title_size:
            # get top-left
            if not title_candidate or phrase['bbox'][0] < title_candidate['bbox'][0]:
                title_candidate = phrase

        # If size is min 1 more than dominant or font is bold o. ä. -> section header
        elif size > dominant_size or any(k in fontname for k in bold_keywords) and len(phrase['text'].split()) <= 7:
            section_headers.append(phrase)

    return title_candidate, section_headers


def split_rows_into_logical_tables(rows):
    logical_tables = []
    current_table = []

    for row in rows:
        if len(row) <= 1:
            continue  # Skip to short rows

        if not current_table:
            current_table.append(row)
            continue

        prev_row = current_table[-1]

        # Check if current row fits with previous row (structure and alignment)
        if rows_are_visually_aligned(prev_row, row) \
                and rows_are_similar(prev_row, row, tolerance=25, min_shared=2, max_cell_diff=1) \
                and columns_are_compatible(prev_row, row, tolerance=40, max_mismatches=1):
            current_table.append(row)
        else:
            if len(current_table) > 1:
                logical_tables.append(current_table)
            current_table = [row] if len(row) > 1 else []

    if len(current_table) > 1:
        logical_tables.append(current_table)

    return logical_tables


def rows_are_visually_aligned(row1, row2, x_overlap_threshold=0.5):
    if not row1 or not row2:
        return False

    overlaps = 0
    for cell1 in row1:
        for cell2 in row2:
            # Compare overlap between cells
            left1, _, right1, _ = cell1['bbox']
            left2, _, right2, _ = cell2['bbox']
            overlap = max(0, min(right1, right2) - max(left1, left2))
            width1 = right1 - left1
            width2 = right2 - left2
            if overlap / max(width1, width2) > x_overlap_threshold:
                overlaps += 1
                break  # Only need one good match per cell

    # at least half the cells ahve to align
    result = overlaps >= min(len(row1), len(row2)) / 2
    return result


def columns_are_compatible(row1, row2, tolerance=40, max_mismatches=1):
    # Compare x-center of each cell to check for column
    x1_centers = [((cell['bbox'][0] + cell['bbox'][2]) / 2) for cell in row1]
    x2_centers = [((cell['bbox'][0] + cell['bbox'][2]) / 2) for cell in row2]

    mismatches = 0
    for xc1 in x1_centers:
        # Check if there is matching column position in row2
        if not any(abs(xc1 - xc2) < tolerance for xc2 in x2_centers):
            mismatches += 1

    result = mismatches <= max_mismatches
    return result


def group_words(words, x_threshold=10, y_threshold=3):
    # Sort words top-to-bottom then left-to-right
    words_sorted = sorted(words, key=lambda w: (round(w['top'], 1), w['x0']))

    lines = []
    current_line = []
    current_top = None

    # Group words in lines based on vertical closeness
    for word in words_sorted:
        if current_line and abs(word['top'] - current_top) > y_threshold:
            lines.append(current_line)
            current_line = []

        current_line.append(word)
        current_top = word['top']

    if current_line:
        lines.append(current_line)

    # group words in phrases by messure horizontal gaps between words
    grouped_phrases = []
    for line in lines:
        phrase = [line[0]]
        for w1, w2 in zip(line, line[1:]):
            gap = w2['x0'] - w1['x1']
            if gap < x_threshold:
                phrase.append(w2)
            else:
                grouped_phrases.append(phrase)
                phrase = [w2]
        grouped_phrases.append(phrase)

    # phrases with bounding boxes and full text
    results = []
    for group in grouped_phrases:
        text = ' '.join([w['text'] for w in group])
        x0 = min(w['x0'] for w in group)
        x1 = max(w['x1'] for w in group)
        top = min(w['top'] for w in group)
        bottom = max(w['bottom'] for w in group)
        results.append({'text': text, 'tokens': group, 'bbox': [x0, top, x1, bottom]})

    return results


def group_into_rows(phrases, y_tolerance=3):
    # Group phrases into rows by comparing y-center
    rows = defaultdict(list)
    for phrase in phrases:
        y_center = (phrase['bbox'][1] + phrase['bbox'][3]) / 2
        key = round(y_center / y_tolerance) * y_tolerance  # Normalize row key
        rows[key].append(phrase)
    # Sort phrases in each row left-to-right
    sorted_phrases = [sorted(row, key=lambda p: p['bbox'][0]) for _, row in sorted(rows.items())]
    return sorted_phrases
