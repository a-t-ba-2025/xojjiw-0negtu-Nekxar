import re
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepTable(AbstractContentPipelineStep):
    def apply(self):
        layout_tables = self.layout_json["Table"]
        for layout_table in layout_tables:
            header_exists = has_a_header(layout_table)
            layout_table["has_header"] = header_exists
            table_json = self.convert_table_to_json(layout_table, has_header=header_exists)
            if self.log:
                print(f"### {header_exists}: Table JSON:\n", table_json)
        return layout_tables

    def convert_table_to_json(self, table, has_header):
        rows = table["rows"]

        if not rows:
            return {}

        if has_header:
            header_row = [cell["text"].strip() for cell in rows[0]]
            data_rows = rows[1:]
        else:
            header_row = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows

        result = {}
        for i, row in enumerate(data_rows):
            row_dict = {}
            for j, cell in enumerate(row):
                col_name = header_row[j] if j < len(header_row) else f"col_{j}"
                row_dict[col_name] = cell["text"].strip()
            result[f"pos_{i + 1}"] = row_dict

        if self.log:
            print(result)
        return result


# Heuristics for detecting table headers, compare two rows cell by cell based on cell_class
def cell_class_similarity(row_a, row_b):
    matches = sum(a["cell_class"] == b["cell_class"] for a, b in zip(row_a, row_b))
    return matches / max(len(row_a), len(row_b))


#Checks symbol differences between two rows
def has_symbol_difference(row_a, row_b):
    differences = sum(a["has_symbol"] != b["has_symbol"] for a, b in zip(row_a, row_b))
    return differences / max(len(row_a), len(row_b))


def is_likely_header(classification_table, similarity_threshold=0.5, symbol_threshold=0.75):
    if len(classification_table) < 2:
        return False

    first_row = classification_table[0]
    data_rows = classification_table[1:]

    # Average similarity of all other rows to the first row
    similarities = [cell_class_similarity(first_row, row) for row in data_rows]
    mean_similarity = sum(similarities) / len(similarities)

    # Average symbol difference to the first row
    symbol_diffs = [has_symbol_difference(first_row, row) for row in data_rows]
    mean_symbol_diff = sum(symbol_diffs) / len(symbol_diffs)

    # If the structure is very different or symbol usage varies significantly -> likely a header
    if mean_similarity < similarity_threshold or mean_symbol_diff > symbol_threshold:
        return True
    return False


#Classifies each cell in table with type and symbol
def classify_table_cells(table):
    classification_table = []

    for row in table["rows"]:
        classified_row = []
        for cell in row:
            classification = classify_cell(cell["text"])
            classified_row.append(classification)
        classification_table.append(classified_row)

    return classification_table


# Assigns type to a cell based on its content
def classify_cell(text: str) -> dict:
    text_clean = text.strip()
    has_symbol = any(sym in text_clean for sym in ['%', '‰', '°'])
    is_money = any(sym in text_clean for sym in ['€', '$', '£', '¥'])

    # Remove common separators for numeric detection
    text_clean = text_clean.replace('.', '').replace(',', '.')

    numeric_part = text_clean.replace(' ', '')

    # Detect float or int
    float_match = re.fullmatch(r"-?\d+\.\d+", numeric_part)
    int_match = re.fullmatch(r"-?\d+", numeric_part)

    if is_money:
        cell_class = "money"
    elif float_match:
        cell_class = "float"
    elif int_match:
        cell_class = "int"
    else:
        cell_class = "string"

    return {"text": text_clean, "cell_class": cell_class, "has_symbol": has_symbol}


def has_a_header(table):
    classification_table = classify_table_cells(table)
    has_table_header_row = is_likely_header(classification_table)
    return has_table_header_row
