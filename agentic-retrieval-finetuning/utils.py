"""
Utilities for PDF loading and image conversion.
"""

import os
import base64
from io import BytesIO
from typing import Optional

from PIL import Image
from pdf2image import convert_from_path
from datasets import load_dataset, disable_progress_bar

# Disable progress bars for cleaner output
disable_progress_bar()


def get_pdf_page_as_png(
    pdf_filename: str, 
    page_number: int, 
    dpi: int = 200,
    dataset_name: str = "agentic-document-ai/agentic-document-ai"
) -> Image.Image:
    """
    Get a specific page from a PDF in the HuggingFace dataset as a PIL Image.
    
    Args:
        pdf_filename: Name of the PDF file (e.g., "document.pdf")
        page_number: Page number (1-indexed)
        dpi: DPI for rendering (default: 200)
        dataset_name: HuggingFace dataset name
        
    Returns:
        PIL Image object
    """
    # Load dataset
    dataset = load_dataset(
        dataset_name,
        data_files="data/documents/*.pdf",
        split="train",
        verification_mode="no_checks"
    )
    
    # Find the PDF by filename
    pdf_path = None
    for item in dataset:
        pdf_obj = None
        if 'file' in item:
            pdf_obj = item['file']
        elif 'pdf' in item:
            pdf_obj = item['pdf']
        else:
            for key, value in item.items():
                if hasattr(value, 'stream') and hasattr(value.stream, 'name'):
                    pdf_obj = value
                    break
        
        if pdf_obj:
            path = pdf_obj.stream.name
            name = os.path.basename(path)
            
            if name == pdf_filename:
                pdf_path = path
                break
    
    if not pdf_path:
        raise ValueError(f"PDF '{pdf_filename}' not found in dataset")
    
    # Convert the specific page to image
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
    if not images:
        raise ValueError(f"Could not extract page {page_number} from {pdf_filename}")
    
    return images[0]


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string (PNG format)."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def resize_image_if_needed(
    image: Image.Image, 
    max_base64_size: int = 4 * 1024 * 1024
) -> tuple[Image.Image, str]:
    """
    Resize image if base64 encoding exceeds max size.
    
    Args:
        image: PIL Image
        max_base64_size: Maximum size in bytes for base64 string
        
    Returns:
        Tuple of (possibly resized image, base64 string)
    """
    base64_image = image_to_base64(image)
    
    scale_factor = 1.0
    while len(base64_image) > max_base64_size:
        scale_factor *= 0.9
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        print(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        base64_image = image_to_base64(image)
    
    return image, base64_image
