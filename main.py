import os
import tempfile # For creating temporary directories
from ocr_utils import (
    process_large_pdf
)
import tempfile # For creating temporary directories
import os

def main():
    # Define your input and output directories
    pdf_input_folder = "pdf_docs"
    final_output_folder = "output" # All final text files go here directly

    # Ensure output base directory exists
    os.makedirs(final_output_folder, exist_ok=True)

    # List all PDF files in the input folder
    pdf_files = [f for f in os.listdir(pdf_input_folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in the '{pdf_input_folder}' directory. Please place your PDFs there.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_name in pdf_files:
        print(f"\n--- Starting processing for '{pdf_name}' ---")

        pdf_path = os.path.join(pdf_input_folder, pdf_name)
        
        # Determine the final output path for this document's text
        output_filename_base = os.path.splitext(pdf_name)[0]
        final_output_path = os.path.join(final_output_folder, f"{output_filename_base}.txt")

        # Skip if the output file already exists (to prevent re-processing)
        if os.path.exists(final_output_path):
            print(f"SKIPPED: Output file '{os.path.basename(final_output_path)}' already exists. Skipping '{pdf_name}'.")
            continue

        # Create a temporary directory for images for the current PDF
        # This ensures unique temporary folders for each PDF and easy cleanup
        try:
            with tempfile.TemporaryDirectory() as temp_images_dir:
                print(f"Using temporary image directory: {temp_images_dir}")

                # Step 1: Process the PDF (converts to images, OCRs in a single call, cleans images)
                extracted_text = process_large_pdf(pdf_path, temp_images_dir)

                # Step 2: Save the final extracted text to the flat output folder
                with open(final_output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"SUCCESS: Final extracted text saved to: {final_output_path}")

        except ValueError as e:
            print(f"SKIPPED: OCR failed for '{pdf_name}' due to content restriction or invalid response: {e}")
            print(f"This PDF was likely flagged for copyrighted or highly sensitive material by the AI model. "
                  f"No output file generated for this document.")
            # No output file created for this PDF in case of ValueError

        except Exception as e: # Catch any other unexpected errors
            print(f"ERROR: An unexpected error occurred while processing '{pdf_name}': {e}")
            print(f"No output file generated for this document due to an unexpected error.")

    print("\n--- All PDF OCR attempts completed. ---")
    print(f"Check the '{final_output_folder}' directory for successfully processed text files.")

if __name__ == "__main__":
    main()
