import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity
from src.pipeline.stepContent.contentStrategy.StrategyContentPipelineSteps.AbstractContentPipelineStep import AbstractContentPipelineStep


class StepCorrector(AbstractContentPipelineStep):
    def __init__(self, image, text_json, layout_json, log: bool = False):
        super().__init__(image=image, text_json=text_json, layout_json=layout_json, log=log)
        self.symspell = None

    def apply(self):
        self.load_symspell()
        corrected_ocr = []

        # valid words only: no digits, no special characters
        valid_word_pattern = re.compile(r"^[A-Za-zÄÖÜäöüß\-]{2,}$")

        for entry in self.text_json:
            original_text = entry["text"]

            # skip masked words from NER-step
            if "[" in original_text and "]" in original_text:
                corrected_ocr.append(entry)
                continue

            words = original_text.split()
            corrected_words = []

            for word in words:
                if valid_word_pattern.fullmatch(word):
                    suggestions = self.symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                    suggestion = self.get_suggestion(word, suggestions)
                    if suggestion is None:
                        corrected_words.append(word)
                        continue

                    #  original upper or lower
                    if word.isupper():
                        suggestion = suggestion.upper()
                    elif word[0].isupper():
                        suggestion = suggestion[0].upper() + suggestion[1:]

                    corrected_words.append(suggestion)
                else:
                    corrected_words.append(word)

            corrected_text = " ".join(corrected_words)
            entry["text"] = corrected_text

            # store original text if correction happened
            if corrected_text != original_text:
                entry["original_text"] = original_text

            corrected_ocr.append(entry)

        return corrected_ocr

    def get_suggestion(self, word, suggestions):
        # Filter suggestions and keep only  terms which showld be corrected
        valid = [s.term for s in suggestions if self.should_correct(word, s.term)]
        first_valid = valid[0] if valid else None
        return first_valid

    def should_correct(self, word: str, suggestion: str) -> bool:
        if word in self.symspell._words or word.lower() in self.symspell._words or word.lower().capitalize() in self.symspell._words:
            return False
        if suggestion not in self.symspell._words:
            return False
        if re.search(r"\d", suggestion):
            return False
        if suggestion.isupper() and not word.isupper():
            return False
        if len(suggestion) < 0.7 * len(word):
            return False
        return True

    def load_symspell(self):  # load symspell dictionary
        dictionary_path = os.getenv("SYM_DICT_PATH")

        # initialize SymSpell
        self.symspell = SymSpell(2, 7)

        # load dictionary file with UTF-8 encoding
        try:
            with open(dictionary_path, encoding="utf-8") as f:
                if not self.symspell.load_dictionary(f, term_index=0, count_index=1):
                    raise FileNotFoundError(f"Could not load SymSpell dictionary at {dictionary_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SymSpell dictionary: {e}")
