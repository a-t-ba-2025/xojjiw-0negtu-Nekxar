import os
from tqdm import tqdm
from pipeline.Pipeline import Pipeline
from dotenv import load_dotenv
import time
import logging
import warnings

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

load_dotenv()  # load .env-File

# environment variables
input_folder = os.getenv('INPUT_PATH')

is_logging = True
is_dev_mode = True
run_preprocessing = True if not is_dev_mode else True
run_text_extraction = True if not is_dev_mode else True
run_layout = True if not is_dev_mode else True
run_content = True if not is_dev_mode else True
run_postprocessor = True if not is_dev_mode else True


def main():
    all_files = os.listdir(input_folder)

    # add specific name(s) of file(s) of the input_folder to array below, otherwise all files of input_folder will be processed
    target_files = [] # for example 'png_1.png'

    if len(target_files) == 0:
        process_files = all_files
    else:
        process_files = [file for file in all_files if file in target_files]

    # Allow only supported file types
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
    valid_files = []

    for file in process_files:
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in allowed_extensions and name:  # name check ensures no files like '.jpeg' only
            valid_files.append(file)
        else:
            print(f"[WARNING] File '{file}' is invalid or has an unsupported extension and will be skipped.")

    if not valid_files:
        print("[WARNING] No valid files found for processing.")
        return

    start_time = time.time()
    for file in tqdm(valid_files):
        print(f"\n")
        with Pipeline(file, log=is_logging, dev_mode=is_dev_mode) as pipeline:
            pipeline.run(
                run_preprocessing=run_preprocessing,
                run_text_extraction=run_text_extraction,
                run_layout=run_layout,
                run_content=run_content,
                run_postprocessor=run_postprocessor
            )

    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
