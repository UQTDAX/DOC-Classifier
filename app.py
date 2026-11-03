import logging
import shutil

from pathlib import Path

from engine.config import config
from engine.text_processor import extract_text_from_file, save_to_file
from engine.pii_remover import setup_presidio_analyzer, propagate_person_names
from engine.text_classifier import load_master_classifications, load_prompt_template, classify_text, save_result

# ============ Setup ============
logger = logging.getLogger(__name__)

def move_to_processed(file_path: str):
    """Move processed file from Pending to Processed folder"""
    try:
        pending_path = Path(config.PENDING_DIR) / Path(file_path).name
        processed_path = Path(config.PROCESSED_DIR) / Path(file_path).name

        if pending_path.exists():
            shutil.move(str(pending_path), str(processed_path))
            logger.info(f"Moved {file_path} to Processed folder")
        else:
            logger.warning(f"File not found in Pending folder: {file_path}")
    except Exception as e:
        logger.error(f"Error moving file {file_path} to Processed: {e}")

# ============ Main Batch Processor ============
def process_batch():
    """Process pending files from queue"""
    Path(config.PENDING_DIR).mkdir(exist_ok=True)
    Path(config.PROCESSED_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)

    # Load master classifications
    master_classifications = load_master_classifications()
    logger.info(f"Master classifications loaded: {master_classifications}")

    # Load prompt template
    prompt_template = load_prompt_template()

    # Get PDF files from Pending folder
    pending_path = Path(config.PENDING_DIR)
    pdf_files = list(pending_path.glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDF files found in Pending folder")
        return

    logger.info(f"Found {len(pdf_files)} files to process")

    analyzer, anonymizer = setup_presidio_analyzer()

    # Process in batches
    for i, pdf_file in enumerate(pdf_files[:config.BATCH_SIZE]):
        logger.info(f"Processing batch item {i+1}/{min(len(pdf_files), config.BATCH_SIZE)}")

        file_text = extract_text_from_file(str(pdf_file))
        save_to_file(str(pdf_file), file_text, "")

        results = analyzer.analyze(text=file_text, language='en')
        results = propagate_person_names(file_text, results)
        anonymized_text = anonymizer.anonymize(text=file_text, analyzer_results=results).text
        save_to_file(str(pdf_file), anonymized_text, "_anonymized")
        
        result = classify_text(master_classifications, prompt_template, anonymized_text, str(pdf_file))
        # save_result(result)

        # # Reload master classifications in case it was updated
        # if result.master_list_updated:
        #     master_classifications = load_master_classifications()
        #     logger.info(f"Master classifications updated. New list: {master_classifications}")

        # # Move file to Processed folder
        # move_to_processed(str(pdf_file))

    logger.info(f"Batch processing complete. Processed {min(len(pdf_files), config.BATCH_SIZE)} files")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.LOG_DIR}/classification.log"),
            logging.StreamHandler()
        ]
    )
    # Presidio logs every recognizer it loads; only show its errors
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    process_batch()

if __name__ == "__main__":
    main()
