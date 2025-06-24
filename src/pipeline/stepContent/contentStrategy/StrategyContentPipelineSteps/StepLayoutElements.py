from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepLayoutElements(AbstractContentPipelineStep):
    def apply(self):
        # This dictionary will store extracted layout elements
        result = {
            "titles": [],
            "section_headers": [],
            "page_headers": []
        }

        # Loop over supported layout categories
        for category, key in [("Title", "titles"),
                              ("Section-header", "section_headers"),
                              ("Page-header", "page_headers")]:
            for block in self.layout_json.get(category, []):
                text_parts = []
                bboxes = []
                confidences = []

                # Collect text, bounding boxes, and confidence values from all cells
                for row in block.get("rows", []):
                    for cell in row:
                        text_parts.append(cell.get("text", "").strip())
                        bboxes.append(cell.get("bbox", []))
                        confidences.append(cell.get("confidence", None))

                # Only store if there is text content
                if text_parts:
                    result[key].append({
                        "category": category,
                        "text": " ".join(text_parts),  # Combine all cell texts
                        "bbox_union": self.merge_bboxes(bboxes),  # Merge all boxes to one surrounding box
                        "confidence_avg": round(sum([c for c in confidences if c]) / len(confidences), 3)
                            if confidences else None  # Average confidence
                    })

        if self.log:
            for key in result:
                print(f"{key.upper()}:")
                for entry in result[key]:
                    print(f"  {entry['text']} → {entry['bbox_union']}")

        return result

    def merge_bboxes(self, bboxes):
        # Merge a list of bounding boxes into one surrounding rectangle
        if not bboxes:
            return []
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return [x0, y0, x1, y1]
