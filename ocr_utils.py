import os
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import shutil # Import shutil for removing directories
import json # Import json for pretty printing
import io # Import io for in-memory byte streams

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Configure the Gemini API with your key
genai.configure(api_key=api_key)

# Initialize the Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Converts each page of a PDF file into a high-resolution image, saving directly as JPEG.

    Args:
        pdf_path (str): The file path of the PDF.
        output_folder (str): The directory to save the output images.
        dpi (int): The resolution (dots per inch) for the output images.

    Returns:
        list: A list of file paths for the created images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- REVISED IMAGE CONVERSION START ---
    # Directly instruct convert_from_path to save as JPEG and return paths
    # This is the most reliable way to ensure the desired format.
    image_paths = convert_from_path(
        pdf_path,
        dpi=dpi,
        output_folder=output_folder, # Tell it where to save the files
        fmt='jpeg',                  # Tell it to save as JPEG
        paths_only=True              # Tell it to return the paths it saved
    )
    # --- REVISED IMAGE CONVERSION END ---

    # Add a debug print to confirm the generated paths are JPEGs
    print(f"  [PDF_DEBUG] pdf2image generated paths (first 5): {[os.path.basename(p) for p in image_paths[:5]]}")

    return image_paths

def batch_images(image_paths, batch_size=25):
    """
    Yields successive n-sized chunks from a list of image paths.
    """
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def ocr_with_gemini(image_paths, instruction_prefix=""):
    """
    Performs OCR on a list of images using Gemini 1.5 Flash.

    Args:
        image_paths (list): A list of file paths for the images to process.
        instruction_prefix (str): An optional prefix for specific instructions.

    Returns:
        str: The extracted text from the images.
    """
    images_for_gemini = []
    for path in image_paths:
        # --- DEBUGGING START ---
        print(f"  [OCR_DEBUG] Attempting to open path for Gemini API: {path}")
        if not os.path.exists(path):
            print(f"  [OCR_DEBUG] WARNING: Image file does not exist at path: {path}. Skipping.")
            continue
        if not path.lower().endswith(('.jpg', '.jpeg', '.png')): # Also check for PNG just in case
            print(f"  [OCR_DEBUG] WARNING: Unexpected image extension '{os.path.splitext(path)[1]}' for path: {path}. Expected .jpg/.png.")
        # --- DEBUGGING END ---

        try:
            img = Image.open(path)
            # Ensure image is in RGB mode, as some APIs prefer it
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images_for_gemini.append(img)
        except Exception as img_err:
            print(f"  [OCR_DEBUG] ERROR: Could not open/process image '{path}': {img_err}. Skipping this image.")
            # We don't raise here, we just skip the problematic image in the batch
            # If all images in a batch fail, images_for_gemini will be empty and handled below.
            continue


    # --- DEBUGGING START ---
    print(f"  [OCR_DEBUG] Processing batch with {len(images_for_gemini)} images (after loading/conversion).")
    if not images_for_gemini:
        print("  [OCR_DEBUG] WARNING: No valid images to send to Gemini in this batch! Returning empty string.")
        return ""
    # --- DEBUGGING END ---

    prompt = f"""
    {instruction_prefix}

    Extract ALL text content from the following document pages.
    Transcribe every single visible character, word, number, and symbol.
    Preserve the original document structure:
    -   Maintain paragraph breaks and line breaks.
    -   Keep headings, subheadings, and lists as they appear.
    -   Convert tables into Markdown format, ensuring correct alignment.
    -   Process multi-column layouts from left-to-right, top-to-bottom.
    -   Extract any legible text from charts, graphs, or images (titles, labels, legends).
    -   Include headers, footers, and page numbers.
    -   DO NOT add any comments, summaries, or extraneous text.
    -   The output should be the complete and exact text content of the document.
    """

    content = [prompt, *images_for_gemini] # Use the prepared images_for_gemini list

    try:
        response = model.generate_content(content)

        # --- DEBUGGING START ---
        print(f"  [OCR_DEBUG] Raw response object (first candidate):")
        if response.candidates:
            first_candidate = response.candidates[0]
            print(f"    Finish Reason: {first_candidate.finish_reason.name if first_candidate.finish_reason else 'N/A'}")
            if first_candidate.safety_ratings:
                print("    Safety Ratings:")
                for rating in first_candidate.safety_ratings:
                    print(f"      - {rating.category.name}: {rating.probability.name}")
            if first_candidate.content:
                print(f"    Content Parts (text): {len([p for p in first_candidate.content.parts if p.text])}")
                text_part_snippet = next((p.text for p in first_candidate.content.parts if p.text), "")
                print(f"    Text Snippet (first 200 chars): {text_part_snippet[:200] if text_part_snippet else 'No text part found'}")
            else:
                print("    No content in first candidate.")
        else:
            print("    No candidates returned in response.")
        # --- DEBUGGING END ---

        return response.text
    except ValueError as e:
        print(f"  [OCR_DEBUG] ERROR: ValueError when trying to get response.text in ocr_with_gemini. Error: {e}")
        raise e
    except Exception as e:
        print(f"  [OCR_DEBUG] UNEXPECTED ERROR: An unexpected error occurred during model generation: {e}")
        raise e

# Rest of ocr_utils.py remains the same:
def ocr_complex_document(image_paths):
    instruction_prefix = """
    **Special emphasis for complex layouts:**
    -   Ensure accurate Markdown table recreation.
    -   Strictly maintain multi-column reading order (left-to-right, top-to-bottom).
    -   Extract all text from charts and graphs.
    """
    return ocr_with_gemini(image_paths, instruction_prefix)

def ocr_financial_document(image_paths):
    instruction_prefix = """
    **Special emphasis for financial documents:**
    -   Achieve 100% numerical accuracy, including decimals and currency symbols.
    -   Precisely transcribe financial tables into Markdown.
    -   Capture all dates and critical sections like footnotes.
    """
    return ocr_with_gemini(image_paths, instruction_prefix)

def verify_ocr_quality(image_path, extracted_text):
    image = Image.open(image_path)
    prompt = f"""
    **Task: OCR Quality Verification**
    You are a quality assurance analyst. I have an original document page image and the text extracted from it via OCR.
    Your job is to compare the image with the text and report any discrepancies.
    **Analyze for:**
    1.  **Missing Text**: Any words, sentences, or paragraphs missing from the text.
    2.  **Incorrect Characters**: Misrecognized letters, numbers, or symbols (e.g., 'l' vs '1', 'O' vs '0').
    3.  **Table Structure Errors**: Misaligned columns or rows in Markdown tables.
    4.  **Formatting Issues**: Lost paragraph breaks or incorrect list formatting.
    **Extracted Text to Verify:**
    ---
    {extracted_text}
    ---
    Provide a summary of any errors found. If no errors are found, respond with "No errors found."
    """
    response = model.generate_content([prompt, image])
    return response.text

def process_large_pdf(pdf_path, temp_images_folder):
    print(f"Converting '{os.path.basename(pdf_path)}' to images...")
    # This call now uses the fixed convert_pdf_to_images which will return JPEG paths
    image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)

    # --- DEBUGGING START ---
    print(f"  [PDF_DEBUG] Temporary images saved to: {temp_images_folder}. Manual inspection is recommended.")
    # --- DEBUGGING END ---

    image_batches = list(batch_images(image_paths, 25))

    full_extracted_text = ""

    for i, batch in enumerate(image_batches):
        print(f"Processing batch {i + 1} of {len(image_batches)} for '{os.path.basename(pdf_path)}' (Pages {i*25 + 1} to {min((i+1)*25, len(image_paths))})...")
        batch_text = ocr_complex_document(batch)
        full_extracted_text += f"\n\n--- END OF BATCH {i + 1} ---\n\n{batch_text}"

    print(f"Cleaning up temporary images for '{os.path.basename(pdf_path)}'...")
    # Remember to uncomment this block once debugging is complete!
    if os.path.exists(temp_images_folder):
        shutil.rmtree(temp_images_folder)
        print(f"Removed temporary image folder: {temp_images_folder}")

    return full_extracted_text

def harmonize_document(extracted_text):
    prompt = f"""
    **Task: Document Harmonization for RAG System - Absolute Fidelity Required**

    The following text was extracted from a single large document in multiple batches. The batches are separated by '--- END OF BATCH ---' markers.

    Your task is to merge this content into a single, seamless, and absolutely coherent document. **The highest priority is to retain EVERY piece of information and maintain the precise original structure, flow, and formatting from the source document.** This output will be used directly for RAG chunking, so any alteration, omission, or addition is highly detrimental.

    **Instructions (Adhere Strictly):**
    1.  **Remove ALL batch separation markers.**
    2.  **Fix formatting breaks and stitch content seamlessly**:
        * Ensure paragraphs, headings, and lists flow naturally across batch boundaries.
        * Correct any unintended line breaks, extra spaces, or missing spaces introduced by the OCR or batching.
    3.  **Stitch broken tables**: If a Markdown table was split across a batch boundary, **merge it back into a single, perfectly formatted Markdown table**. Verify that all column headers and row data are correctly aligned and complete.
    4.  **Preserve all original formatting elements**: Maintain bolding, italics (where clear from context or markdown), and other structural cues that aid readability and context for RAG.
    5.  **DO NOT summarize, paraphrase, interpret, add comments, or generate new information.** Your function is purely to assemble the transcribed text into a single, correct, and continuous representation of the original document.
    6.  **The output MUST be the complete, raw, merged text content, nothing more.**
    """
    content = [prompt, extracted_text]
    response = model.generate_content(content)
    return response.text
