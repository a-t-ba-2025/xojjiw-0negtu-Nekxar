from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepCorrector import StepCorrector
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepDemasking import StepDemasking
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepFlairNER import StepFlairNER
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepLayoutElements import StepLayoutElements
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepMasking import StepMasking
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepRegex import StepRegex
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.StepTable import StepTable


class StrategyContentPipeline:
    def __init__(self, image, text_json, layout_json, log: bool = False):
        self.result_layout_elements = None
        self.result_regex = None
        self.result_flair_ner = None
        self.result_table = None
        self.demasked_text_json = None
        self.text_json_corrected = None
        self.mask_map = None
        self.masked_text_json = None
        self.image = image
        self.text_json = text_json
        self.layout_json = layout_json
        self.log = log

    def __enter__(self):
        print(f"## [Pipeline] [ContextContent] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"## [Pipeline] [ContextContent] [{self.__class__.__name__}] completed")

    def execute(self):
        with StepFlairNER(image=self.image, text_json=self.text_json, layout_json=self.layout_json, log=self.log) as step_flair_ner:
            self.result_flair_ner = step_flair_ner.apply()
        with StepRegex(image=self.image, text_json=self.text_json, layout_json=self.layout_json, log=self.log) as step_regex:
            self.result_regex = step_regex.apply()
        with StepMasking(image=self.image, text_json=self.text_json, layout_json=self.layout_json, log=self.log, ner_results=self.result_flair_ner, regex_results=self.result_regex) as step_masking:
            self.masked_text_json, mask_map = step_masking.apply()
        with StepCorrector(image=self.image, text_json=self.masked_text_json, layout_json=self.layout_json, log=self.log) as step_corrector:
            self.text_json_corrected = step_corrector.apply()
        with StepDemasking(text_json_corrected=self.text_json_corrected, mask_map=mask_map) as step_demasking:
            self.demasked_text_json = step_demasking.apply()
        with StepTable(image=self.image, text_json=self.demasked_text_json, layout_json=self.layout_json, log=self.log) as step_table:
            self.result_table = step_table.apply()
        with StepLayoutElements(image=self.image, text_json=self.demasked_text_json, layout_json=self.layout_json, log=self.log) as step_layout:
            self.result_layout_elements = step_layout.apply()
        return {
            "named_entities": self.result_flair_ner,
            "regex_matches": self.result_regex,
            "text_corrected": self.demasked_text_json,
            "tables": self.result_table,
            "other_elements": self.result_layout_elements
        }
