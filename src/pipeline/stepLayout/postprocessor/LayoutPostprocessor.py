import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


class LayoutPostprocessor:
    def __init__(self, text_json: list, log: bool = False):
        self.text_json = text_json
        self.log = log

    def run(self, layout_boxes: list):
        print(f"## [Pipeline] [ContextLayout] [{self.__class__.__name__}] started")
        label_map = [  # List of labels and score thresholds
            ("Table", 0.7),
            ("Caption", 0.5),
            ("Footnote", 0.5),
            ("Caption", 0.5),
            ("Footnote", 0.5),
            ("Formula", 0.5),
            ("List-item", 0.5),
            ("Page-footer", 0.5),
            ("Page-header", 0.5),
            # ("Picture", 0.7),
            ("Section-header", 0.5),
            ("Text", 0.5),
            ("Title", 0.5),
        ]
        results = {}
        for label, filterscore in label_map:
            label_results = self.process_layout_category(
                layout_boxes,
                filter_name=label,
                filter_score=filterscore,
            )
            results[label] = []  # new List for ecah Label
            for i in label_results:
                results[label].append(label_results[i])

        for label in results:
            if label == "Table":
                results[label] = process_tables(results["Table"]) or []
            else:
                results[label] = process_elements(results[label], label) or []

        unmatched = self.filter_unmatched_ocr(results)
        results["unmatched"] = unmatched

        if self.log:
            for label in results:
                print(label)
                for entry in results[label]:
                    print(f"\t{entry}")
        return results

    def filter_unmatched_ocr(self, results):
        used_in_results = set()

        for label_type, label_items in results.items():
            for item in label_items:  # Iterate over all text rows in layout element
                for row in item.get("rows", []):
                    for cell in row:
                        # Extract text and bounding box from the OCR cell
                        bbox = tuple(cell.get("bbox"))
                        text = cell.get("text")
                        if bbox and text:  # Only add if both text and bbox are there
                            used_in_results.add((text, bbox))

        unmatched = [
            ocr for ocr in self.text_json
            if (ocr["text"], tuple(ocr["bbox"])) not in used_in_results
        ]

        return unmatched

    def process_layout_category(self, layout_boxes, filter_name, filter_score=0.7):
        # Get layout elements filtered by label and the score
        filtered = filter_elements(layout_boxes, filter_name=filter_name, filter_score=filter_score)
        matches = self.match_ocr_to_layout(filtered)
        return matches

    def match_ocr_to_layout(self, layout_boxes: list):
        # Match OCR boxes to layout boxes by their overlapping area
        matches = {}

        for i, layout in enumerate(layout_boxes):
            x1_l, y1_l, x2_l, y2_l = layout["box"]

            matched_ocr = []

            for ocr in self.text_json:
                if "bbox" not in ocr:
                    if self.log:
                        print(f"[WARN] OCR-element without box: {ocr}")
                    continue

                x1_o, y1_o, x2_o, y2_o = ocr["bbox"]

                inter_x1 = max(x1_l, x1_o)
                inter_y1 = max(y1_l, y1_o)
                inter_x2 = min(x2_l, x2_o)
                inter_y2 = min(y2_l, y2_o)

                inter_width = max(0, inter_x2 - inter_x1)
                inter_height = max(0, inter_y2 - inter_y1)
                inter_area = inter_width * inter_height

                ocr_area = (x2_o - x1_o) * (y2_o - y1_o)

                if ocr_area > 0 and (inter_area / ocr_area) > 0.5:
                    matched_ocr.append(ocr)

            matches[i] = {
                "layout_box": layout,
                "ocr_matches": matched_ocr
            }

        return matches


###############################################
########### Table-Layout-Elements #############
###############################################

# splits Tables into smaller tables if tehy have different structures
# returns cleaned and filtered tables with bounding boxes.
def process_tables(tables):  # tables (with OCR results)
    final_tables = []

    for key, table in enumerate(tables):
        if "ocr_matches" not in table:
            continue  # skip if there is no OCR data

        # Sort OCR results (top to bottom a. left to right)
        sorted_ocr = sort_ocr_entries(table["ocr_matches"])

        # Group text into rows based on vertical positions
        rows = group_ocr_into_rows(sorted_ocr)

        # Split the table if the structure of the rows changes too much
        subtables = split_table_on_structure_change(rows)

        for i, subtable_rows in enumerate(subtables):
            if len(subtable_rows) < 2:
                continue  # ignore very small tables

            # Combine all cells in this subtable
            flat_cells = [cell for row in subtable_rows for cell in row]

            # Create a bounding box around all the cells
            bbox = create_bounding_box(flat_cells)
            layout_box = table.get("layout_box", {})
            score = layout_box.get("score", 0.0)
            # Save the result
            final_tables.append({
                "rows": subtable_rows,
                "bbox": bbox,
                "score": score,
            })

    # Remove tables that are inside other tables (maybe detected twice)
    return remove_nested_tables(final_tables)


# Checks if the structure of the table changes a lot between rows
# If yes, it splits them into different subtables
def split_table_on_structure_change(rows, tolerance=25, min_shared=2, max_cell_diff=1):
    if not rows:
        return []

    tables = []
    current_table = [rows[0]]

    for i in range(1, len(rows)):
        prev_row = current_table[-1]
        current_row = rows[i]

        # Check if current row is very different from the previous one
        if not rows_are_similar(prev_row, current_row, tolerance, min_shared, max_cell_diff):
            tables.append(current_table)
            current_table = [current_row]
        else:
            current_table.append(current_row)

    if current_table:
        tables.append(current_table)

    return tables


# Checks if one box lies mostly inside another
def is_mostly_inside(inner, outer):
    # Unpack coordinates of the inner box
    x1, y1, x2, y2 = inner

    # Unpack coordinates of the outer box
    ox1, oy1, ox2, oy2 = outer

    # Calculate coordinates of the overlapping area (intersection)
    inter_x1 = max(x1, ox1)  # left edge of overlap
    inter_y1 = max(y1, oy1)  # top edge of overlap
    inter_x2 = min(x2, ox2)  # right edge of overlap
    inter_y2 = min(y2, oy2)  # bottom edge of overlap

    # Calculate the area of the overlapping part
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of the inner box
    inner_area = (x2 - x1) * (y2 - y1)

    # Check how much of the inner box lies inside the outer box
    at_least_threshold_inside = inter_area / inner_area >= 0.8
    return at_least_threshold_inside


# Removes tables that are nested inside others
def remove_nested_tables(tables):
    to_remove = set()
    # Compare every table with every other tables
    for i, t1 in enumerate(tables):
        for j, t2 in enumerate(tables):
            if i == j:
                continue
            # Check if table t1 is far inside table t2
            if is_mostly_inside(t1["bbox"], t2["bbox"]):
                # If t1 has a smaller confidence, mark for remove
                if t1["score"] < t2["score"]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    # tables that are not marked for removal
    not_remove_tables = [t for idx, t in enumerate(tables) if idx not in to_remove]
    return not_remove_tables


# Checks how similar two rows are, -> alignment of cells
def rows_are_similar(row1, row2, tolerance=25, min_shared=2, max_cell_diff=0):
    def x_refs(row):
        return {
            "left": sorted([ocr["bbox"][0] for ocr in row]),
            "center": sorted([(ocr["bbox"][0] + ocr["bbox"][2]) // 2 for ocr in row]),
            "right": sorted([ocr["bbox"][2] for ocr in row])
        }

    def count_matches(xs1, xs2):
        return sum(any(abs(x1 - x2) <= tolerance for x2 in xs2) for x1 in xs1)

    refs1, refs2 = x_refs(row1), x_refs(row2)
    matches = 0

    # Compare positions of cells
    for align in ["left", "center", "right"]:
        matches += count_matches(refs1[align], refs2[align])

    # Check if the number of aligned positions is enough and the row sizes are similar
    min_cells = min(len(row1), len(row2))
    cell_diff = abs(len(row1) - len(row2))

    return matches >= min_shared and cell_diff <= max_cell_diff


###############################################
########### Other Layout-Elements #############
###############################################
def process_elements(elements, label_name):
    result_elements = []

    # Loop over each detected layout element
    for key, entry in enumerate(elements):
        if "ocr_matches" not in entry:
            continue  # skip if there is no OCR data

        # If OCR exist sort them
        sorted_ocr = sort_ocr_entries(entry["ocr_matches"])
        # group the sorted OCR results in rows
        rows = group_ocr_into_rows(sorted_ocr)
        # Skip if no text rows
        if len(rows) == 0:
            continue

        bbox = create_bounding_box(sorted_ocr)

        # Add result
        result_elements.append({
            "rows": rows,  # grouped text rows
            "bbox": bbox,  # position of layout element
        })

    # Return all elements
    return result_elements


###############################################
########### OCR Utility Functions #############
###############################################

def create_bounding_box(cells):
    if not cells:
        return [0, 0, 0, 0]
    x1 = min(cell["bbox"][0] for cell in cells)
    y1 = min(cell["bbox"][1] for cell in cells)
    x2 = max(cell["bbox"][2] for cell in cells)
    y2 = max(cell["bbox"][3] for cell in cells)
    return [x1, y1, x2, y2]


# sorts OCR entries in order top-to-bottom and left-to-right
def sort_ocr_entries(ocr_entries, y_threshold=10):
    if not ocr_entries:
        return []

    # sort by vertical position top to botto
    ocr_entries = sorted(ocr_entries, key=lambda x: x["bbox"][1])
    grouped = []  # list of lines
    current_line = []  # current line

    for entry in ocr_entries:
        y = entry["bbox"][1]  # top y-position of box

        if not current_line:
            current_line.append(entry)  # first
        else:
            prev_y = current_line[-1]["bbox"][1]
            # Check if entry is close to previous y -> probably saame line
            if abs(y - prev_y) <= y_threshold:
                current_line.append(entry)
            else:
                # Sort line from left to right
                grouped.append(sorted(current_line, key=lambda x: x["bbox"][0]))
                current_line = [entry]  # start new line

    # last line
    if current_line:
        grouped.append(sorted(current_line, key=lambda x: x["bbox"][0]))

    # single list
    return [cell for row in grouped for cell in row]


# groups OCR boxes in rows via clustering
def group_ocr_into_rows(ocr_matches: list, eps: int = 15):
    if not ocr_matches:
        return []

    # vertical center of each box
    y_centers = [(ocr["bbox"][1] + ocr["bbox"][3]) / 2 for ocr in ocr_matches]
    y_array = np.array(y_centers).reshape(-1, 1)

    # DBSCAN: find groups of similar y-centers -> rows
    clustering = DBSCAN(eps=eps, min_samples=1).fit(y_array)
    labels = clustering.labels_

    #OCR-bbox to row
    rows = {}
    for label, ocr in zip(labels, ocr_matches):
        rows.setdefault(label, []).append(ocr)

    sorted_rows = []
    for row_ocr in rows.values():
        # Sort boxes in row from left to right
        row_sorted = sorted(row_ocr, key=lambda x: x["bbox"][0])
        sorted_rows.append(row_sorted)

    # sort all rows from top to bottom
    sorted_rows.sort(key=lambda r: np.mean([ocr["bbox"][1] for ocr in r]))

    return sorted_rows


def filter_elements(layout_boxes: list, filter_name, filter_score=0.7):
    # Keep only the boxes with the right label and good enough score
    filtered_elements = []
    for box in layout_boxes:
        if box["label_name"] != filter_name:
            continue
        if box["score"] < filter_score:
            continue
        filtered_elements.append(box)
    return filtered_elements
