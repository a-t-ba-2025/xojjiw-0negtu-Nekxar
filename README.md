# Context-Aware Content Analysis Pipeline

This project implements a modular pipeline for extracting, analyzing, and transforming content from unstructured documents. It was developed as part of a Bachelor's thesis in Computer Science.

---

## Project Structure

```
src
├── main.py
├── util.py
├── pipeline/
    ├── AbstractContext.py
    ├── Pipeline.py
    ├── stepFiletype/
    │   ├── FiletypeDeterminer.py
    ├── stepPreprocessing/
    │   ├── ContextPreprocessor.py
    │   └── preprocessStrategy/
    │       ├── StrategyPreProcessPipeline.py
    │       └── strategyPreProcessPipelineSteps/
    │           ├── AbstractPreprocessPipelineStep.py
    │           ├── StepBinarize.py
    │           ├── StepContrast.py
    │           ├── StepDenoise.py
    │           ├── StepDeskew.py
    │           ├── StepGrayscale.py
    │           ├── StepShadow.py
    ├── stepTextExtraction/
    │   ├── ContextTextExtraction.py
    │   └── textExtractionStrategy/
    │       ├── AbstractStrategyTextExtraction.py
    │       ├── StrategyPaddle.py
    │       ├── StrategyPdf.py
    │       ├── StrategyTesseract.py
    ├── stepLayout/
    │   ├── ContextLayout.py
    │   ├── layoutStrategy/
    │   │   ├── AbstractStrategyLayout.py
    │   │   ├── StrategyDETR.py
    │   │   ├── StrategyFRCNN.py
    │   │   ├── StrategyHybridFRCNN_DETR.py
    │   │   ├── StrategyPDF.py
    │   └── postprocessor/
    │       ├── LayoutPostprocessor.py
    ├── stepContent/
    │   ├── ContextContent.py
    │   └── contentStrategy/
    │       ├── StrategyContentPipeline.py
    │       └── StrategyContentPipelineSteps/
    │           ├── AbstractContentPipelineStep.py
    │           ├── StepCorrector.py
    │           ├── StepDemasking.py
    │           ├── StepFlairNER.py
    │           ├── StepLayoutElements.py
    │           ├── StepMasking.py
    │           ├── StepRegex.py
    │           ├── StepTable.py
    └── stepPostProcessing/
        └── PostProcessor.py
```

---

## Execution

```bash
python src/main.py
```

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

The `main.py` file is the entry point of the project. It loads all available files from the `INPUT_PATH` directory (as configured via `.env`) and processes them through the pipeline. Alternatively, a list of specific files can be passed to restrict the execution to selected documents.

---


## Configuration

The `.env` file can be used to configure paths, models, debug modes, and more.

```ini
# OCR Configuration
TESSERACT_CMD=path\to\tesseract.exe

# Model paths
TEXT_MODEL=path/to/text_models
FASTERCRNN_LAYOUT=path/to/layout_models/FasterRCNN
DETR_LAYOUT=path/to/layout_models/DETR
DETR_PROCESSOR=path/to/layout_models/DETR-processor
FLAIR_CONTENT=path/to/content_models/Flair/ner-german-large.pt

# Resources
SYM_DICT_PATH=./resources/de_symspell_dict.txt

# Input/output data paths
INPUT_PATH=data/input
PREPROCESSED_PATH=data/preprocessed_images
TEXT_IMAGE_PATH=data/text/images
TEXT_JSON_PATH=data/text/json
LAYOUT_IMAGE_PATH=data/layout/images
LAYOUT_JSON_PATH=data/layout/layout_json
CONTENT_JSON_PATH=data/content/content_json
OUTPUT_PATH=data/output
```

Each variable defines:
- `TESSERACT_CMD`: Path to your local Tesseract OCR executable
- `TEXT_MODEL`: Path to text-model for identifying text-rich or layout-rich documents
- `FASTERCRNN_LAYOUT`: Path to layout-model Faster-R-CNN 
- `DETR_LAYOUT`: Path to layout-model DETR
- `DETR_PROCESSOR`  Path to DETR-Processor
- `FLAIR_CONTENT`: Path to text-model for identifying text-rich or layout-rich documents
- `SYM_DICT_PATH`: Path to German dictionary for typo correction
- `*_PATH`: Defines input, intermediate, and output locations for pipeline steps

---

## License

This project was developed as part of a university thesis. It is intended for non-commercial, academic use only.

---

## Author
Bachelor's thesis  
by Anna M. T.
Last update: June 2025
