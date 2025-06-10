# Document OCR Pipeline with Gemini 2.0 Flash

This repository contains a powerful Optical Character Recognition (OCR) pipeline built with Python and the Google Gemini 1.5 Flash model. It's designed to process PDF documents, extract detailed text, and prepare it for applications like Retrieval Augmented Generation (RAG), focusing on high fidelity and structural preservation.

Note: I just vibecoded this using Gemini 2.5 Pro

## Project Structure

The project is organized for clarity and efficient processing:

```
your_ocr_project/
├── .env                  # Stores your Google API key (ignored by Git for security)
├── main.py               # The main script to run the OCR pipeline
├── ocr_utils.py          # Contains core OCR functions and model interactions
├── pdf_docs/             # Place your input PDF documents here
└── output/               # Where the final extracted text files will be saved (ignored by Git)
```

## Getting Started

Follow these steps to set up and run the OCR pipeline on your machine:

### 1. Install Dependencies

You'll need a few Python libraries and the `poppler-utils` system package to convert PDFs into images.

First, install `poppler-utils`:

```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

Next, install the required Python libraries:

```bash
pip install google-generativeai pypdf pdf2image Pillow python-dotenv
```

### 2. Obtain Your Gemini API Key

To use the Gemini 2.0 Flash model, you need an API key:

1. Visit Google AI Studio.
2. Create a new API key.

### 3. Configure Your API Key

Create a file named `.env` in the root directory of your project (i.e., `your_ocr_project/.env`). Add your API key to this file:

```
GOOGLE_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with the actual key you obtained.

### 4. Place Your Documents

Put all the PDF documents you want to process into the `pdf_docs/` folder.

### 5. Run the OCR Pipeline

Navigate to the project's root directory in your terminal and execute the main script:

```bash
python main.py
```

The script will process each PDF, print progress messages, and save the extracted text files directly into the `output/` folder. Temporary image files generated during processing will be automatically cleaned up.
