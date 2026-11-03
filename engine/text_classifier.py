import re
import json
import logging
import requests

from jinja2 import Environment, StrictUndefined
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple

from engine.config import config
from engine.text_processor import save_to_file

logger = logging.getLogger(__name__)

@dataclass
class Classification:
    doc_type: str
    confidence: float
    is_new: bool = False  # True if this classification was not in the master list

@dataclass
class ClassificationResult:
    file_path: str
    classifications: List[Classification]
    extracted_text: str
    processing_time: float
    error: Optional[str] = None
    master_list_updated: bool = False  # True if master list was updated during processing

# ============ Master Classifications ============
def load_master_classifications() -> List[str]:
    """Load master classifications from newline-separated text file"""
    config_path = Path(config.CLASSIFICATIONS_FILE)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                classifications = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(classifications)} master classifications")
                return classifications
        except Exception as e:
            logger.error(f"Failed to load master classifications: {e}")
            return []
    else:
        logger.warning(f"Master classifications file not found: {config.CLASSIFICATIONS_FILE}")
        return []

def save_master_classifications(classifications: List[str]):
    """Save master classifications to newline-separated text file"""
    try:
        with open(config.CLASSIFICATIONS_FILE, 'w') as f:
            for classification in classifications:
                f.write(classification + '\n')
        logger.info(f"Updated master classifications with {len(classifications)} entries")
    except Exception as e:
        logger.error(f"Failed to save master classifications: {e}")

def update_master_classifications(new_classifications: List[str]) -> Tuple[List[str], bool]:
    """Update master classifications list with any new ones found

    Returns:
        Tuple of (updated list, whether any new classifications were added)
    """
    current = load_master_classifications()
    updated = False

    for classification in new_classifications:
        if classification not in current:
            current.append(classification)
            updated = True
            logger.info(f"Added new classification to master list: {classification}")

    if updated:
        save_master_classifications(current)

    return current, updated

def load_prompt_template() -> str:
    """Load the classification prompt template from the config folder"""
    prompt_path = Path(config.PROMPT_FILE)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {config.PROMPT_FILE}")

    with open(prompt_path, 'r') as f:
        return f.read()

def setup_prompt(master_classifications: List[str], text: str, prompt_template: str) -> str:
    """Setup the prompt for LLM classification by rendering the Jinja template"""
    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(prompt_template)
    return template.render(classifications=master_classifications, text=text)

# ============ Call LLM for Classification ============
def call_llm(master_classifications: List[str], prompt: str) -> List[Classification]:
    """Call local LLM for multi-classification using master list as context

    Args:
        master_classifications: List of known classification types
        prompt: Formatted prompt for the LLM

    Returns:
        List of Classification objects with is_new flag indicating if classification is in master list
    """
    try:
        response = requests.post(
            config.LLM_API_URL,
            json={
                "model": config.MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": config.TEMPERATURE
            },
            timeout=config.TIMEOUT_SECONDS
        )
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "").strip()

        # Parse JSON response - extract JSON array
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            classifications_data = json.loads(json_match.group())
            classifications = []

            for item in classifications_data:
                doc_type = item.get("type", "Unknown")
                confidence = float(item.get("confidence", 0.0))
                is_new = doc_type not in master_classifications

                classifications.append(
                    Classification(
                        doc_type=doc_type,
                        confidence=confidence,
                        is_new=is_new
                    )
                )

            # Sort by confidence descending
            classifications.sort(key=lambda x: x.confidence, reverse=True)
            return classifications

        return [Classification(doc_type="Unknown", confidence=0.0, is_new=False)]

    except requests.exceptions.Timeout:
        logger.error(f"LLM timeout for classification")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise

# ============ Classify Text ============
def classify_text(master_classifications: List[str], prompt_template: str, file_text: str, file_path: str) -> ClassificationResult:
    """Classify text extracted from a single file

    Args:
        master_classifications: Current list of known classifications
        prompt_template: Prompt template loaded from the config folder        
        file_text: Text extracted from the file
        file_path: Path to the file being processed
    """
    start_time = datetime.now()

    try:
        logger.info(f"Classifying text")

        prompt = setup_prompt(master_classifications, file_text[:config.MAX_CHARACTERS], prompt_template)

        save_to_file(file_path, prompt, "_prompt")
       
        return # Debugging: Stop here to inspect the prompt
    
        # Classify using master list as context
        classifications = call_llm(master_classifications, prompt)

        # Check if any new classifications were found
        new_classifications = [c.doc_type for c in classifications if c.is_new]
        master_list_updated = False

        if new_classifications:
            logger.info(f"Found new classifications: {new_classifications}")
            master_classifications, master_list_updated = update_master_classifications(
                master_classifications + new_classifications
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        result = ClassificationResult(
            file_path=file_path,
            classifications=classifications,
            extracted_text=file_text,
            processing_time=processing_time,
            master_list_updated=master_list_updated
        )

        conf_str = ", ".join([f"{c.doc_type} ({c.confidence:.2f}) {'[NEW]' if c.is_new else '[KNOWN]'}" for c in classifications])
        logger.info(f"Classified {file_path}: {conf_str} in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return ClassificationResult(
            file_path=file_path,
            classifications=[],
            extracted_text=file_text,
            processing_time=(datetime.now() - start_time).total_seconds(),
            error=str(e)
        )

def save_result(result: ClassificationResult):
    """Save result to text file with CLASSIFICATION | CONFIDENCE | STATUS format"""
    output_filename = Path(result.file_path).stem + "_result.txt"
    output_file = Path(config.RESULTS_DIR) / output_filename

    with open(output_file, 'w') as f:
        f.write("CLASSIFICATION | CONFIDENCE | STATUS\n")
        for classification in result.classifications:
            status = "NEW" if classification.is_new else "KNOWN"
            f.write(f"{classification.doc_type} | {classification.confidence:.2f} | {status}\n")

    logger.info(f"Results saved to: {output_file}")
