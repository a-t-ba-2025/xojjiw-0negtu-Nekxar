import re
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepRegex(AbstractContentPipelineStep):
    def apply(self):
        result = self.extract()
        return result

    def extract(self):
        # regex patterns for different data types
        patterns = {
            "IBAN": re.compile(r'^[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?$'),
            "Email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "Datum": re.compile(r'^(?:\d{1,2}[./-]){2}\d{2,4}$'),
        }

        result = {key: [] for key in patterns}

        for entry in self.text_json:
            text = entry.get("text", "")
            bbox = entry.get("bbox", [])

            for key, pattern in patterns.items():
                match = pattern.search(text)
                if not match:
                    continue

                # remove spaces for standardization
                value = match.group().replace(" ", "")

                # save result with text and bounding box
                result[key].append({
                    "text": value,
                    "bbox": bbox
                })
        if self.log:
            print(result)
        return result
