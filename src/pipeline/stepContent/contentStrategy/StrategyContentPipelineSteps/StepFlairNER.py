import os
from dotenv import load_dotenv
from flair.data import Sentence
from flair.models import SequenceTagger
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep

load_dotenv()


class StepFlairNER(AbstractContentPipelineStep):
    def apply(self):
        model_path = os.getenv('FLAIR_CONTENT')
        tagger = SequenceTagger.load(model_path)
        ocr_data = self.text_json

        ner_results = []

        for entry in ocr_data:
            text = entry["text"]

            if not text.strip():
                continue

            sentence = Sentence(text)
            tagger.predict(sentence)

            for entity in sentence.get_spans("ner"):
                ner_results.append({
                    "entity": entity.text,
                    "label": entity.get_label("ner").value,
                    "score": round(entity.score, 3),
                })

        if self.log:
            for result in ner_results:
                print(result)
        return ner_results
