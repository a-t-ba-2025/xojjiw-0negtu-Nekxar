from collections import defaultdict
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepMasking(AbstractContentPipelineStep):
    def __init__(self, image, text_json, layout_json, log: bool = False, ner_results=None, regex_results=None):
        super().__init__(image=image, text_json=text_json, layout_json=layout_json, log=log)
        self.mask_map = None
        self.ner_results = ner_results or []
        self.regex_results = regex_results or {}

    def apply(self):
        counters = defaultdict(int)
        mask_map = {}
        masked_json = []

        # Combine NER and RegEx
        to_mask = []

        for ner in self.ner_results:
            to_mask.append((ner["entity"], ner["label"].upper()))

        for category, entries in self.regex_results.items():
            for match in entries:
                to_mask.append((match["text"], category.upper()))

        # Iterate all OCR
        for entry in self.text_json:
            text = entry["text"]
            for original, label in to_mask:
                if original in text:
                    counters[label] += 1
                    placeholder = f"[{label}_{counters[label]}]"
                    text = text.replace(original, placeholder)
                    mask_map[placeholder] = original
            entry["text"] = text
            masked_json.append(entry)

        if self.log:
            print("### Masking completed:")
            for k, v in mask_map.items():
                print(f"  {k} -> {v}")

        self.mask_map = mask_map
        return masked_json, mask_map
