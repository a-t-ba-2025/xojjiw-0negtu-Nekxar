import os
import json
import cv2 as cv
from dotenv import load_dotenv
from src.util import save_image, save_json, save_file
from .stepFiletype.FiletypeDeterminer import FiletypeDeterminer
from .stepPostProcessing.PostProcessor import PostProcessor
from .stepPreprocessing.ContextPreprocessor import ContextPreprocessor
from .stepTextExtraction.ContextTextExtraction import ContextTextExtraction
from .stepLayout.ContextLayout import ContextLayout
from .stepContent.ContextContent import ContextContent

load_dotenv()  # Load environment variables from .env file

class Pipeline:
    def __init__(self, upload_file, log=False, dev_mode=False):
        self.upload_file = upload_file
        self.log = log
        self.dev_mode = dev_mode

        # Filetyping Results
        self.file_name = None
        self.file_type = None
        self.input_path = None
        self.typed_file = None
        self.is_mostly_text = None

        # Preprocessing Results
        self.preprocessed_image = None

        # Text Extraction Results
        self.text_image = None
        self.text_json = None
        self.words = None

        # Layout Analysis Results
        self.layout_image = None
        self.layout_json = None

        # Content Analysis Results
        self.content_json = None

    def __enter__(self):
        print(f"[Pipeline] started: {self.upload_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"[Pipeline] completed: {self.upload_file}")

    def run(self, run_preprocessing, run_text_extraction, run_layout, run_content, run_postprocessor):
        self.run_file_determining()
        self.load_or_run_preprocessing(run_preprocessing)
        self.load_or_run_text_extraction(run_text_extraction)
        self.load_or_run_layout(run_layout)
        self.load_or_run_content(run_content)
        self.load_or_run_postprocessor(run_postprocessor)

    def run_file_determining(self):
        # Check if required data exist
        if self.upload_file is None:
            raise ValueError("Cannot run FiletypeDeterminer: required input 'upload_file' is missing")

        # Detect file type and get relevant input data
        with FiletypeDeterminer(upload_file=self.upload_file, log=self.log) as filetype_determiner:
            self.file_name, self.file_type, self.input_path, self.typed_file, self.is_mostly_text = filetype_determiner.run()

    def load_or_run_preprocessing(self, run_step: bool = False):
        if not run_step:
            # If Step ist not executed: Try to load preprocessed image from cache
            path = os.path.join(os.getenv("PREPROCESSED_PATH"), f"{self.file_name}.png")
            if os.path.exists(path):
                self.preprocessed_image = cv.imread(path)
                return
            else:
                if self.log:
                    print(f"[WARNING] Preprocessed image not found: {path}, run preprocessing step")

        if run_step:
            # Check if required data exist
            if self.file_type is None or self.typed_file is None:
                raise TypeError("Cannot run preprocessing: file_type and/or typed_file is missing")
            # Run preprocessing
            with ContextPreprocessor(file_type=self.file_type, typed_file=self.typed_file, log=self.log) as step:
                preprocessed_images = step.run()
                self.preprocessed_image = preprocessed_images[-1] if preprocessed_images else None # use only image last in array, others for debugging
                # Save result only in dev mode
                if self.dev_mode and self.preprocessed_image is not None:
                    save_image(self.preprocessed_image, save_dir="PREPROCESSED_PATH", filename=self.file_name)

    def load_or_run_text_extraction(self, run_step: bool = False):
        if not run_step:
            img_path = os.path.join(os.getenv("TEXT_IMAGE_PATH"), f"{self.file_name}.png")
            json_path = os.path.join(os.getenv("TEXT_JSON_PATH"), f"{self.file_name}.json")
            # Try to load text data from cache
            if os.path.exists(img_path):
                self.text_image = cv.imread(img_path)
            else:
                if self.log:
                    print(f"[WARNING] OCR image not found: {img_path}, run text extraction step")

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    self.text_json = json.load(f)
            else:
                if self.log:
                    print(f"[WARNING] OCR JSON not found: {json_path}, run text extraction step")

        if run_step:
            # Check if required data exist
            if (self.file_type is None
                    or (self.file_type != "pdf" and self.preprocessed_image is None)
                    or (self.file_type == "pdf" and self.input_path is None)):
                raise TypeError("Cannot run text extraction: required inputs are missing depending on file_type")
            # Run text extraction
            with ContextTextExtraction(file_type=self.file_type, image=self.preprocessed_image, is_mostly_text=self.is_mostly_text,
                                       pdf_path=self.input_path[0], log=self.log) as step:
                self.text_image, self.text_json, self.words = step.run()
                # if dev_mode: Save result
                if self.dev_mode:
                    if self.text_image is not None:
                        if self.file_type == "pdf":
                            save_file(self.text_image, save_dir="TEXT_IMAGE_PATH", filename=self.file_name)
                        else:
                            save_image(self.text_image, save_dir="TEXT_IMAGE_PATH", filename=self.file_name)
                    if self.text_json is not None:
                        save_json(self.text_json, save_dir="TEXT_JSON_PATH", filename=self.file_name)

    def load_or_run_layout(self, run_step: bool = False):
        if not run_step:
            path_image = os.path.join(os.getenv("LAYOUT_IMAGE_PATH"), f"{self.file_name}.png")
            json_path = os.path.join(os.getenv("LAYOUT_JSON_PATH"), f"{self.file_name}.json")

            # Try to load layout data from cache
            if os.path.exists(path_image):
                self.layout_image = cv.imread(path_image)
            else:
                if self.log:
                    print(f"[WARNING] Layout image not found: {path_image}, run layout analysis step")

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    self.layout_json = json.load(f)
            else:
                if self.log:
                    print(f"[WARNING] Layout JSON not found: {json_path}, run layout analysis step")

        if run_step:
            # Check if required data exist
            if self.file_type is None or self.typed_file is None or self.input_path is None or self.text_json is None:
                raise TypeError("Cannot run layout step: file_type and/or typed_file and/or  input_path and/or text_json is missing")
            # Run layout detection
            with ContextLayout(file_type=self.file_type, text_json=self.text_json, words=self.words, image=self.typed_file,
                               pdf_path=self.input_path[0], log=self.log) as step:
                self.layout_image, self.layout_json = step.run()
                # Save result if in dev mode
                if self.dev_mode:
                    if self.layout_image is not None:
                        save_image(self.layout_image, save_dir="LAYOUT_IMAGE_PATH", filename=self.file_name)
                    if self.layout_json is not None:
                        save_json(self.layout_json, save_dir="LAYOUT_JSON_PATH", filename=self.file_name)

    def load_or_run_content(self, run_step: bool = False):
        if not run_step:
            content_path = os.path.join(os.getenv("CONTENT_JSON_PATH"), f"{self.file_name}.json")
            if os.path.exists(content_path):
                with open(content_path, "r", encoding="utf-8") as f:
                    self.content_json = json.load(f)
                return
            else:
                if self.log:
                    print(f"[WARNING] CONTENT JSON not found: {content_path}, run Content Analysis step")

        if run_step:
            # Check if required data exist
            if self.typed_file is None or self.text_json is None or self.layout_json is None:
                raise TypeError("Cannot run content analysis: typed_file and/or text_json and/or layout_json is missing")
            # Run content analysis
            with ContextContent(image=self.typed_file, text_json=self.text_json,
                                layout_json=self.layout_json, log=self.log) as step:
                self.content_json = step.run()
                if self.dev_mode and self.content_json is not None:
                    save_json(self.content_json, save_dir="CONTENT_JSON_PATH", filename=self.file_name)

    def load_or_run_postprocessor(self, run_step: bool = False):
        if not run_step:
            return
        # Check if required data exist
        if self.file_name is None or self.content_json is None:
            raise TypeError("Cannot run postprocessor: file_name and/or content_json is missing")
        # Post processing to final JSON
        with PostProcessor(file_name=self.file_name, content_json=self.content_json, log=self.log) as step:
            result_json = step.run()
            if self.log:
                print(result_json)
            if result_json is not None:
                save_json(result_json, save_dir="OUTPUT_PATH", filename=self.file_name)
