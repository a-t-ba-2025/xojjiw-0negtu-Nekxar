import re
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepDemasking(AbstractContentPipelineStep):
    def __init__(self, text_json_corrected, mask_map, image=None, text_json=None, layout_json=None, log=False):
        super().__init__(image=image, text_json=text_json, layout_json=layout_json, log=log)
        self.text_json_corrected = text_json_corrected
        self.mask_map = mask_map

    def apply(self):
        demasked_json = []

        # Regex for Placeholder set in StepMasking [...]
        pattern = re.compile(r"\[[A-Z]+_\d+\]")

        for entry in self.text_json_corrected:
            text = entry["text"]

            # Replace all masks in text with original text
            def replace_match(match):
                token = match.group(0)
                return self.mask_map.get(token, token)  # replace if in map

            new_text = pattern.sub(replace_match, text)
            entry["text"] = new_text
            demasked_json.append(entry)

        if self.log:
            print(demasked_json)

        return demasked_json
