from datetime import datetime


class PostProcessor:
    def __init__(self, file_name, content_json, log: bool = False):
        self.file_name = file_name
        self.content_json = content_json
        self.log = log

    def __enter__(self):
        print(f"# [Pipeline][{self.__class__.__name__}] started: {self.file_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"# [Pipeline][{self.__class__.__name__}] completed: {self.file_name}")

    def run(self):
        semantic = {
            "document_id": self.file_name,
            "entities": [],
            "blocks": [],
            "metadata": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "pipeline_version": "v1.0"
            }
        }

        seen_blocks = set()

        def add_block(block):
            text = block.get("text", None)
            bbox = tuple(block["bbox"])
            key = (block["type"], text, bbox)
            if key not in seen_blocks:
                semantic["blocks"].append(block)
                seen_blocks.add(key)

        # Named Entities
        for ent in self.content_json.get("named_entities", []):
            semantic["entities"].append({
                "type": ent.get("label"),
                "text": ent.get("entity"),
                "confidence": ent.get("score", 1.0),
                "source": "flair_ner"
            })

        # Regex Matches
        for label, matches in self.content_json.get("regex_matches", {}).items():
            for match in matches:
                semantic["entities"].append({
                    "type": label,
                    "text": match.get("text"),
                    "confidence": 1.0,
                    "source": "regex"
                })

        # Textblocks
        for entry in self.content_json.get("text_corrected", []):
            add_block({
                "type": "text",
                "text": entry.get("text"),
                "bbox": entry.get("bbox"),
                "confidence": entry.get("confidence"),
            })

        # Tables
        for table in self.content_json.get("tables", []):
            rows = table.get("rows", [])
            if not rows:
                continue
            has_header = table.get("has_header", True)
            if has_header:
                header = [cell.get("text") for cell in rows[0]]
                data_rows = [[cell.get("text") for cell in row] for row in rows[1:]]
            else:
                header = [f"col_{i}" for i in range(len(rows[0]))]
                data_rows = [[cell.get("text") for cell in row] for row in rows]

            add_block({
                "type": "table",
                "header": header,
                "rows": data_rows,
                "id": table.get("id"),
                "bbox": table.get("bbox"),
                "score": table.get("score"),
                "has_header": has_header
            })

        # Layout (not text or table)
        excluded = {"text", "table"}
        for key, entries in self.content_json.get("other_elements", {}).items():
            block_type = key.rstrip("s").lower()
            if block_type in excluded:
                continue
            for entry in entries:
                add_block({
                    "type": block_type,
                    "text": entry.get("text"),
                    "bbox": entry.get("bbox_union"),
                    "confidence": entry.get("confidence_avg"),
                    "source": "layout"
                })

        # sort blocks
        sort_order = {
            "title": 0,
            "section_header": 1,
            "page_header": 2,
            "text": 3,
            "table": 4,
            "footnote": 5
        }
        semantic["blocks"].sort(key=lambda b: sort_order.get(b["type"], 99))

        return semantic
