import os
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import shutil # Import shutil for removing directories

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Configure the Gemini API with your key
genai.configure(api_key=api_key)

# Initialize the Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Converts each page of a PDF file into a high-resolution image.

    Args:
        pdf_path (str): The file path of the PDF.
        output_folder (str): The directory to save the output images.
        dpi (int): The resolution (dots per inch) for the output images.

    Returns:
        list: A list of file paths for the created images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # This function does the conversion
    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)

    return image_paths

def batch_images(image_paths, batch_size=25):
    """
    Yields successive n-sized chunks from a list of image paths.
    """
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def ocr_with_gemini(image_paths, instruction):
    """
    Performs OCR on a list of images using Gemini 2.0 Flash.

    Args:
        image_paths (list): A list of file paths for the images to process.
        instruction (str): The prompt/instruction for the model.

    Returns:
        str: The extracted text from the images.
    """
    images = [Image.open(path) for path in image_paths]

    # --- REVISED PROMPT START ---
    prompt = f"""
    {instruction}

    You are an **extremely meticulous and literal OCR engine**. Your sole purpose is to **transcribe every single visible character and piece of text** from the provided document pages. This output is for a RAG (Retrieval Augmented Generation) system, so **completeness and absolute fidelity to the source content are paramount**.

    **Strict Adherence Guidelines (Do NOT deviate):**
    1.  **NO SUMMARIZATION, NO INTERPRETATION, NO ADDITIONS, NO DELETIONS, NO MODIFICATIONS.**
    2.  **Transcribe EXACTLY what you see.** Every word, every number, every symbol, every punctuation mark.
    3.  **Preserve Original Structure:**
        * **Paragraphs:** Maintain original paragraph breaks.
        * **Headings/Subheadings:** Include all headings and subheadings as they appear.
        * **Lists:** Replicate bullet points, numbered lists, and other list formats precisely.
        * **Indentation:** Preserve indentation where it conveys structural meaning (e.g., in outlines or code snippets, though less common in reports).
    4.  **Tables:** Convert tables into **perfectly formatted Markdown tables**.
        * Every column header and row data point must be correctly aligned.
        * Transcribe all numerical and text data within tables with **100% accuracy, including decimals and currency symbols.**
    5.  **Multi-Column Layouts:** Process text in multi-column layouts **strictly sequentially from left-to-right, then top-to-bottom**. Do not intermix text from different columns on the same line. Use newlines to separate logical blocks.
    6.  **Charts, Graphs, Images (Text Extraction):** Extract all **legible text** from these elements: titles, captions, axis labels, legends, data labels, and any other embedded text. Do not describe the image itself, only extract its text content.
    7.  **Headers, Footers, Page Numbers, Footnotes:** Include these as they appear. They are part of the document's content.
    8.  **Whitespace:** Preserve significant whitespace that impacts structure (e.g., line breaks, spacing between paragraphs). Avoid excessive or unnecessary newlines.
    9.  **No Commentary:** Do not generate any conversational text, introductory phrases ("Here is the extracted text:"), or concluding remarks. Just the raw, transcribed document content.

    **Focus on capturing the entirety of the document's textual information for precise retrieval.**
    """
    # --- REVISED PROMPT END ---

    # Prepare the content for the API call
    content = [prompt, *images]

    response = model.generate_content(content)
    return response.text

# Retaining specialized OCR functions, though main.py will use the general one with a specific instruction
def ocr_complex_document(image_paths):
    """
    Uses a specialized prompt for OCR on documents with complex layouts.
    This function will be called by process_large_pdf with an explicit instruction.
    """
    # The instruction here is now more specific, but the main detail is in ocr_with_gemini's base prompt
    instruction = """
    **Task: High-Fidelity Text Extraction for Complex Layouts (RAG Optimization)**

    Focus on capturing all textual information while faithfully representing the original document's structure for optimal RAG chunking.
    """
    return ocr_with_gemini(image_paths, instruction)

def ocr_financial_document(image_paths):
    """
    Uses a highly specific prompt for extracting data from financial documents.
    """
    instruction = """
    **Task: Financial Document OCR for RAG System**

    Extract text from these financial reports with extreme precision, focusing on numerical and structured data, for a RAG system.

    **Key Focus Areas (in addition to general strict guidelines):**
    1.  **Numerical Accuracy**: All numbers, decimals, and percentages must be perfectly transcribed.
    2.  **Currency Symbols**: Correctly associate symbols ($, €, £, etc.) with their corresponding values.
    3.  **Financial Tables**: Replicate the exact structure of all tables (e.g., balance sheets, income statements) using Markdown.
    4.  **Critical Sections**: Pay special attention to cash flow statements, footnotes, and disclosures.
    5.  **Dates**: Accurately capture all financial period dates.
    6.  **Maintain Context**: Ensure that numbers are clearly associated with their labels/descriptions.
    """
    return ocr_with_gemini(image_paths, instruction)

def verify_ocr_quality(image_path, extracted_text):
    """
    Uses Gemini to compare an image with its extracted text to find errors.
    """
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
    """
    Processes a very large PDF by converting it to images, running OCR in batches,
    and returns the combined raw text. Cleans up temporary images.

    Args:
        pdf_path (str): The file path of the PDF.
        temp_images_folder (str): Temporary directory to save images.

    Returns:
        str: The full raw extracted text.
    """
    # Step 1: Convert the entire PDF to images
    print(f"Converting '{os.path.basename(pdf_path)}' to images...")
    image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)

    # Step 2: Create batches of images
    image_batches = list(batch_images(image_paths, 25)) # Adjust batch size as needed

    full_extracted_text = ""
    # Store candidates and safety info for debugging
    all_candidates_info = []

    for i, batch in enumerate(image_batches):
        print(f"Processing batch {i + 1} of {len(image_batches)} for '{os.path.basename(pdf_path)}' (Pages {i*25 + 1} to {min((i+1)*25, len(image_paths))})...")
        try:
            # We'll use the ocr_complex_document function here which calls ocr_with_gemini
            # This ensures the new detailed prompt is always used.
            batch_text = ocr_complex_document(batch)
            full_extracted_text += f"\n\n--- END OF BATCH {i + 1} ---\n\n{batch_text}"
        except Exception as e:
            # Catching the exception here to provide more context about the problematic batch/pages
            print(f"ERROR: Failed to OCR batch {i + 1} (Pages {i*25 + 1} to {min((i+1)*25, len(image_paths))}) for '{os.path.basename(pdf_path)}'.")
            print(f"Error details: {e}")
            # If the error is from response.text, it means no valid parts were returned.
            # We need to re-raise this specific error as it's critical for content safety.
            raise # Re-raise the exception to be caught by main.py

    # Step 3: Clean up temporary images
    print(f"Cleaning up temporary images for '{os.path.basename(pdf_path)}'...")
    if os.path.exists(temp_images_folder):
        shutil.rmtree(temp_images_folder)
        print(f"Removed temporary image folder: {temp_images_folder}")

    return full_extracted_text

def harmonize_document(extracted_text):
    """
    Cleans and harmonizes text extracted in batches.
    """
    # --- REVISED PROMPT START ---
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
    # --- REVISED PROMPT END ---

    # Use the model to process the harmonization prompt
    response = model.generate_content(prompt)
    return response.text