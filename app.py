import os
import sys
import base64
import json
import re
from flask import Flask, render_template, request, jsonify
import pdfplumber
import PyPDF2
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Use /tmp for uploads on Vercel (serverless), otherwise use uploads folder
if os.environ.get('VERCEL'):
    app.config['UPLOAD_FOLDER'] = '/tmp'
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        openai_client = None
        print(f"Warning: Failed to initialize OpenAI client: {str(e)}")
else:
    openai_client = None
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Image processing configuration (memory optimization)
IMAGE_DPI = int(os.getenv('IMAGE_DPI', '72'))  # Reduced from 150 to 72
MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION', '2048'))  # Max width or height in pixels
MAX_IMAGE_SIZE_MB = float(os.getenv('MAX_IMAGE_SIZE_MB', '2.0'))  # Max size per image after compression
MAX_IMAGE_SIZE_BYTES = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
MAX_IMAGES_PER_PAGE = int(os.getenv('MAX_IMAGES_PER_PAGE', '5'))  # Max images per PDF page
MAX_IMAGES_PER_PDF = int(os.getenv('MAX_IMAGES_PER_PDF', '20'))  # Max total images per PDF
MAX_TOTAL_IMAGES_PER_REQUEST = int(os.getenv('MAX_TOTAL_IMAGES_PER_REQUEST', '50'))  # Max images across all files
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '85'))  # JPEG compression quality (1-100)


def optimize_image(pil_image):
    """
    Optimize image for memory efficiency: resize if needed, convert to JPEG, compress.
    Returns (optimized_image_bytes, mime_type) or None if optimization fails.
    """
    try:
        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
            pil_image = rgb_image
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if dimensions exceed maximum
        width, height = pil_image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Compress to JPEG with quality setting
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
        img_bytes = img_buffer.getvalue()
        
        # Check if image size exceeds limit
        if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
            # Try reducing quality progressively
            for quality in [75, 65, 55, 45]:
                img_buffer = BytesIO()
                pil_image.save(img_buffer, format='JPEG', quality=quality, optimize=True)
                img_bytes = img_buffer.getvalue()
                if len(img_bytes) <= MAX_IMAGE_SIZE_BYTES:
                    break
            
            # If still too large, resize further
            if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
                current_width, current_height = pil_image.size
                scale_factor = (MAX_IMAGE_SIZE_BYTES / len(img_bytes)) ** 0.5
                new_width = max(100, int(current_width * scale_factor))
                new_height = max(100, int(current_height * scale_factor))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_buffer = BytesIO()
                pil_image.save(img_buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                img_bytes = img_buffer.getvalue()
        
        return img_bytes, 'image/jpeg'
    
    except Exception as e:
        print(f"Error optimizing image: {str(e)}")
        return None, None


def extract_pdf_content(pdf_path):
    """
    Extract full content from PDF including text, images, and metadata.
    Returns a dictionary with page-by-page content.
    """
    result = {
        'pages': [],
        'metadata': {
            'page_count': 0,
            'title': '',
            'author': '',
            'subject': '',
        }
    }
    
    # Extract text and metadata using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        result['metadata']['page_count'] = len(pdf.pages)
        
        # Get metadata
        if pdf.metadata:
            result['metadata']['title'] = pdf.metadata.get('Title', '')
            result['metadata']['author'] = pdf.metadata.get('Author', '')
            result['metadata']['subject'] = pdf.metadata.get('Subject', '')
        
        # Extract text and images from each page
        total_images_extracted = 0
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text() or ''
            
            # Extract images from pdfplumber (with limits)
            images = []
            if page.images and total_images_extracted < MAX_IMAGES_PER_PDF:
                for img_index, img_obj in enumerate(page.images):
                    # Limit images per page
                    if len(images) >= MAX_IMAGES_PER_PAGE:
                        break
                    
                    # Limit total images per PDF
                    if total_images_extracted >= MAX_IMAGES_PER_PDF:
                        break
                    
                    try:
                        # Get image bounding box
                        bbox = (img_obj.get('x0', 0), img_obj.get('top', 0), 
                                img_obj.get('x1', 0), img_obj.get('bottom', 0))
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            cropped = page.crop(bbox)
                            
                            # Try to get image as PIL Image using pdfplumber's to_image
                            try:
                                # Use reduced DPI for memory efficiency
                                pil_image = cropped.to_image(resolution=IMAGE_DPI)
                                
                                # Optimize image (resize, compress, convert to JPEG)
                                img_bytes, mime_type = optimize_image(pil_image)
                                
                                if img_bytes and mime_type:
                                    # Convert to base64
                                    image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                    images.append({
                                        'index': img_index,
                                        'data': f"data:{mime_type};base64,{image_base64}",
                                        'ext': 'jpg'
                                    })
                                    total_images_extracted += 1
                                    
                                    # Explicitly delete large objects to free memory
                                    del pil_image
                                    del img_bytes
                                else:
                                    print(f"Warning: Failed to optimize image {img_index} from page {page_num}")
                                
                            except MemoryError as mem_error:
                                print(f"Memory error extracting image {img_index} from page {page_num}: {mem_error}")
                                break  # Stop processing images on this page if out of memory
                            except Exception as img_error:
                                print(f"Error converting image {img_index} from page {page_num} to PIL: {img_error}")
                    except MemoryError as mem_error:
                        print(f"Memory error processing image {img_index} from page {page_num}: {mem_error}")
                        break
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num}: {e}")
            
            # Get page dimensions
            width = page.width
            height = page.height
            
            result['pages'].append({
                'page_number': page_num + 1,
                'text': text,
                'images': images,
                'width': width,
                'height': height
            })
    
    # Also try to extract images using PyPDF2 as fallback (with optimization)
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                # Skip if we've already hit the limit
                if page_num < len(result['pages']):
                    current_page_images = len(result['pages'][page_num]['images'])
                    if current_page_images >= MAX_IMAGES_PER_PAGE:
                        continue
                
                try:
                    if '/Resources' in page and '/XObject' in page['/Resources']:
                        xobjects = page['/Resources']['/XObject']
                        if hasattr(xobjects, 'get_object'):
                            xobjects = xobjects.get_object()
                        
                        for obj_name, obj in xobjects.items():
                            # Check limits
                            if page_num < len(result['pages']):
                                if len(result['pages'][page_num]['images']) >= MAX_IMAGES_PER_PAGE:
                                    break
                                if total_images_extracted >= MAX_IMAGES_PER_PDF:
                                    break
                            
                            if hasattr(obj, 'get') and obj.get('/Subtype') == '/Image':
                                try:
                                    # Extract image data
                                    data = obj.get_data()
                                    
                                    # Check raw data size before processing
                                    if len(data) > MAX_IMAGE_SIZE_BYTES * 3:  # Allow 3x before compression
                                        print(f"Skipping very large image from PyPDF2 on page {page_num} ({len(data)} bytes)")
                                        continue
                                    
                                    # Try to load and optimize the image
                                    try:
                                        img_buffer = BytesIO(data)
                                        pil_image = Image.open(img_buffer)
                                        img_bytes, mime_type = optimize_image(pil_image)
                                        
                                        if img_bytes and mime_type:
                                            # Convert to base64
                                            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                            
                                            # Add to images
                                            if page_num < len(result['pages']):
                                                new_index = len(result['pages'][page_num]['images'])
                                                result['pages'][page_num]['images'].append({
                                                    'index': new_index,
                                                    'data': f"data:{mime_type};base64,{image_base64}",
                                                    'ext': 'jpg'
                                                })
                                                total_images_extracted += 1
                                                
                                                # Clean up
                                                del pil_image
                                                del img_bytes
                                        else:
                                            # Fallback: use original data if optimization fails
                                            image_base64 = base64.b64encode(data).decode('utf-8')
                                            ext = 'jpg' if '/DCTDecode' in str(obj.get('/Filter', '')) else 'png'
                                            if page_num < len(result['pages']):
                                                new_index = len(result['pages'][page_num]['images'])
                                                result['pages'][page_num]['images'].append({
                                                    'index': new_index,
                                                    'data': f"data:image/{ext};base64,{image_base64}",
                                                    'ext': ext
                                                })
                                                total_images_extracted += 1
                                    except Exception as img_process_error:
                                        # Fallback: use original data
                                        image_base64 = base64.b64encode(data).decode('utf-8')
                                        ext = 'jpg' if '/DCTDecode' in str(obj.get('/Filter', '')) else 'png'
                                        if page_num < len(result['pages']):
                                            new_index = len(result['pages'][page_num]['images'])
                                            result['pages'][page_num]['images'].append({
                                                'index': new_index,
                                                'data': f"data:image/{ext};base64,{image_base64}",
                                                'ext': ext
                                            })
                                            total_images_extracted += 1
                                    
                                except MemoryError as mem_error:
                                    print(f"Memory error extracting PyPDF2 image from page {page_num}: {mem_error}")
                                    break
                                except Exception as e:
                                    print(f"Error extracting image with PyPDF2 from page {page_num}: {e}")
                except MemoryError as mem_error:
                    print(f"Memory error processing page {page_num} with PyPDF2: {mem_error}")
                    break
                except Exception as e:
                    print(f"Error processing page {page_num} with PyPDF2: {e}")
    except MemoryError as mem_error:
        print(f"Memory error in PyPDF2 image extraction: {mem_error}")
    except Exception as e:
        print(f"Error in PyPDF2 image extraction: {e}")
    
    return result


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


def extract_image_content(image_path):
    """
    Extract content from an image file with memory optimization.
    Returns a dictionary with image data.
    """
    try:
        with Image.open(image_path) as img:
            # Get original image dimensions
            original_width, original_height = img.size
            
            # Optimize image (resize if needed, compress, convert to JPEG)
            img_bytes, mime_type = optimize_image(img.copy())
            
            if not img_bytes or not mime_type:
                raise Exception('Failed to optimize image')
            
            # Get final dimensions after optimization
            optimized_img = Image.open(BytesIO(img_bytes))
            final_width, final_height = optimized_img.size
            optimized_img.close()
            
            # Convert to base64
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Check size limit
            if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
                print(f"Warning: Image {os.path.basename(image_path)} size ({len(img_bytes)} bytes) exceeds limit ({MAX_IMAGE_SIZE_BYTES} bytes)")
            
            return {
                'type': 'image',
                'filename': os.path.basename(image_path),
                'width': final_width,
                'height': final_height,
                'format': 'JPEG',  # Always JPEG after optimization
                'data': f"data:{mime_type};base64,{image_base64}",
                'size': len(img_bytes)
            }
    except MemoryError as mem_error:
        raise Exception(f'Memory error processing image: {str(mem_error)}')
    except Exception as e:
        raise Exception(f'Error processing image: {str(e)}')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extraction (PDFs and images)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        filename_lower = file.filename.lower()
        is_pdf = filename_lower.endswith('.pdf')
        is_image = any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'])
        
        if not (is_pdf or is_image):
            return jsonify({'error': 'Invalid file type. Please upload a PDF or image file.'}), 400
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            if is_pdf:
                # Extract PDF content
                extracted_content = extract_pdf_content(file_path)
                extracted_content['type'] = 'pdf'
                extracted_content['filename'] = file.filename
            else:
                # Extract image content
                extracted_content = extract_image_content(file_path)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify(extracted_content), 200
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


def identify_document_source(filename):
    """Map filename to source type."""
    filename_lower = filename.lower()
    if 'fnol' in filename_lower:
        return 'fnol'
    elif 'claimant' in filename_lower:
        return 'claimant'
    elif 'other_driver' in filename_lower or 'other driver' in filename_lower:
        return 'other_driver'
    elif 'police' in filename_lower:
        return 'police'
    elif 'repair' in filename_lower or 'estimate' in filename_lower:
        return 'repair_estimate'
    elif 'policy' in filename_lower or 'policy_document' in filename_lower:
        return 'policy'
    else:
        return 'unknown'


def normalize_facts(facts_list):
    """Unify conflicting facts into standard schema."""
    normalized_facts = []
    
    for fact in facts_list:
        normalized = fact.copy()
        
        # Normalize directions
        if 'normalized_value' in fact and fact.get('category') == 'location':
            direction = fact.get('normalized_value', '').lower()
            direction_mapping = {
                'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W',
                'northeast': 'NE', 'northwest': 'NW',
                'southeast': 'SE', 'southwest': 'SW'
            }
            for key, value in direction_mapping.items():
                if key in direction:
                    normalized['normalized_value'] = value
                    break
        
        # Normalize impact points
        if fact.get('category') == 'impact':
            impact = fact.get('normalized_value', '').lower()
            impact = impact.replace('-', '_').replace(' ', '_')
            normalized['normalized_value'] = impact
        
        # Normalize time formats
        if fact.get('category') == 'temporal':
            time_value = fact.get('normalized_value', '')
            # Try to extract and normalize time format
            time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', time_value)
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2)
                period = time_match.group(3) or ''
                if period:
                    normalized['normalized_value'] = f"{hour:02d}:{minute} {period.upper()}"
                else:
                    normalized['normalized_value'] = f"{hour:02d}:{minute}"
        
        normalized_facts.append(normalized)
    
    return normalized_facts


def detect_conflicts(facts_list):
    """Identify contradictions across sources using OpenAI."""
    if not openai_client:
        # Fallback: return empty conflicts if OpenAI is not available
        return []
    
    if not facts_list or len(facts_list) < 2:
        # Need at least 2 facts to have conflicts
        return []
    
    # Build system prompt for conflict detection
    system_prompt = """You are an expert auto insurance claims analyst specializing in identifying contradictions and conflicts in claim documents. Your task is to analyze a fact matrix extracted from multiple claim documents and identify all conflicts, contradictions, and inconsistencies.

Your responsibilities:

1. Identify direct contradictions:
   - Different values for the same fact from different sources (e.g., different directions, times, locations)
   - Conflicting statements about the same event
   - Inconsistent normalized values for the same category of fact

2. Identify inconsistencies in related facts:
   - Impact points that don't align with described vehicle movements
   - Temporal facts that conflict with each other
   - Location descriptions that contradict movement patterns
   - Environmental conditions that conflict with other facts

3. Identify implied conflicts:
   - Statements that logically contradict each other
   - Facts that cannot both be true simultaneously
   - Source credibility issues (e.g., police report contradicts driver statement)

4. Analyze conflicts across all source types:
   - fnol (First Notice of Loss)
   - claimant (Claimant statement)
   - other_driver (Other driver statement)
   - police (Police report)
   - repair_estimate (Repair estimate)

For each conflict identified, provide:
- fact_description: A clear description of what fact is in conflict
- sources: Array of source types that contain conflicting information
- conflicting_values: Array of the different values/statements that conflict
- conflict_type: Type of conflict (e.g., 'direct_contradiction', 'inconsistency', 'implied_conflict', 'temporal_conflict', 'location_conflict', 'movement_conflict')
- severity: "high", "medium", or "low" based on the impact on claim assessment
- explanation: A brief explanation of why this is a conflict
- recommended_version: Which conflicting value is more likely to be true (one of the values from conflicting_values array)
- evidence: Detailed explanation of why the recommended version is more likely to be true, including credibility of sources, consistency with other facts, and any supporting evidence
- value_details: For each conflicting value, provide:
  - value: The conflicting value
  - sources: Array of source types that support this value
  - source_snippets: Array of source text snippets (from source_text field) that support this value, one snippet per source

Return your response as a JSON object with this exact structure:
{
  "conflicts": [
    {
      "fact_description": "string",
      "sources": ["array of source strings"],
      "conflicting_values": ["array of conflicting value strings"],
      "conflict_type": "string",
      "severity": "high|medium|low",
      "explanation": "string",
      "recommended_version": "string (one of the conflicting_values)",
      "evidence": "string - detailed explanation of why recommended_version is more likely",
      "value_details": [
        {
          "value": "string (one of the conflicting_values)",
          "sources": ["array of source strings supporting this value"],
          "source_snippets": ["array of source text snippets from the fact matrix"]
        }
      ]
    }
  ]
}

Analyze the following fact matrix:"""
    
    # Format facts for the prompt
    facts_text = "\n\nFact Matrix:\n"
    for idx, fact in enumerate(facts_list):
        facts_text += f"\nFact {idx + 1}:\n"
        facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
        facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
        facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
        facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
        facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
        if fact.get('normalized_value'):
            facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        if fact.get('is_implied'):
            facts_text += f"  Is Implied: {fact.get('is_implied')}\n"
    
    # Prepare content for OpenAI
    full_prompt = system_prompt + facts_text
    
    # Call OpenAI API with JSON mode
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            conflicts = result.get('conflicts', [])
            
            # Ensure backward compatibility: map to expected format
            # Frontend expects: fact_description, sources, conflicting_values
            # Additional fields are optional enhancements
            formatted_conflicts = []
            for conflict in conflicts:
                # Match source snippets to actual facts from the fact matrix
                value_details = conflict.get('value_details', [])
                enhanced_value_details = []
                
                for value_detail in value_details:
                    value = value_detail.get('value', '')
                    detail_sources = value_detail.get('sources', [])
                    ai_snippets = value_detail.get('source_snippets', [])
                    
                    # Find matching facts from the fact matrix
                    matching_facts = []
                    for fact in facts_list:
                        fact_source = fact.get('source', '')
                        fact_value = fact.get('normalized_value', '') or fact.get('extracted_fact', '')
                        
                        # Check if this fact matches the value and source
                        if fact_source in detail_sources:
                            # Check if the fact's value matches (fuzzy match)
                            if value.lower() in fact_value.lower() or fact_value.lower() in value.lower():
                                matching_facts.append(fact)
                    
                    # Use AI-provided snippets if available, otherwise use source_text from matching facts
                    final_snippets = []
                    if ai_snippets:
                        final_snippets = ai_snippets
                    else:
                        # Fallback: use source_text from matching facts
                        for fact in matching_facts[:3]:  # Limit to 3 snippets
                            snippet = fact.get('source_text', '')
                            if snippet and snippet not in final_snippets:
                                final_snippets.append(snippet[:200])  # Limit snippet length
                    
                    enhanced_value_details.append({
                        'value': value,
                        'sources': detail_sources,
                        'source_snippets': final_snippets
                    })
                
                formatted_conflicts.append({
                    'fact_description': conflict.get('fact_description', ''),
                    'sources': conflict.get('sources', []),
                    'conflicting_values': conflict.get('conflicting_values', []),
                    'conflict_type': conflict.get('conflict_type', ''),
                    'severity': conflict.get('severity', 'medium'),
                    'explanation': conflict.get('explanation', ''),
                    'recommended_version': conflict.get('recommended_version', ''),
                    'evidence': conflict.get('evidence', ''),
                    'value_details': enhanced_value_details
                })
            
            return formatted_conflicts
        
        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                conflicts = result.get('conflicts', [])
                formatted_conflicts = []
                for conflict in conflicts:
                    # Match source snippets to actual facts from the fact matrix
                    value_details = conflict.get('value_details', [])
                    enhanced_value_details = []
                    
                    for value_detail in value_details:
                        value = value_detail.get('value', '')
                        detail_sources = value_detail.get('sources', [])
                        ai_snippets = value_detail.get('source_snippets', [])
                        
                        # Find matching facts from the fact matrix
                        matching_facts = []
                        for fact in facts_list:
                            fact_source = fact.get('source', '')
                            fact_value = fact.get('normalized_value', '') or fact.get('extracted_fact', '')
                            
                            # Check if this fact matches the value and source
                            if fact_source in detail_sources:
                                # Check if the fact's value matches (fuzzy match)
                                if value.lower() in fact_value.lower() or fact_value.lower() in value.lower():
                                    matching_facts.append(fact)
                        
                        # Use AI-provided snippets if available, otherwise use source_text from matching facts
                        final_snippets = []
                        if ai_snippets:
                            final_snippets = ai_snippets
                        else:
                            # Fallback: use source_text from matching facts
                            for fact in matching_facts[:3]:  # Limit to 3 snippets
                                snippet = fact.get('source_text', '')
                                if snippet and snippet not in final_snippets:
                                    final_snippets.append(snippet[:200])  # Limit snippet length
                        
                        enhanced_value_details.append({
                            'value': value,
                            'sources': detail_sources,
                            'source_snippets': final_snippets
                        })
                    
                    formatted_conflicts.append({
                        'fact_description': conflict.get('fact_description', ''),
                        'sources': conflict.get('sources', []),
                        'conflicting_values': conflict.get('conflicting_values', []),
                        'conflict_type': conflict.get('conflict_type', ''),
                        'severity': conflict.get('severity', 'medium'),
                        'explanation': conflict.get('explanation', ''),
                        'recommended_version': conflict.get('recommended_version', ''),
                        'evidence': conflict.get('evidence', ''),
                        'value_details': enhanced_value_details
                    })
                return formatted_conflicts
            else:
                # If JSON parsing fails completely, return empty conflicts
                print(f"Warning: Failed to parse conflict detection JSON response: {str(e)}")
                return []
    
    except Exception as e:
        # If OpenAI API fails, return empty conflicts
        print(f"Warning: OpenAI conflict detection failed: {str(e)}")
        return []


def extract_facts_from_documents(files_data):
    """Main extraction orchestrator."""
    all_facts = []
    
    # Prepare content for OpenAI
    content_parts = []
    images_added_count = 0  # Track number of images being sent to OpenAI
    
    text_content = """You are an expert at extracting structured facts from auto insurance claim narratives. 
Analyze the following documents and extract ALL accident-relevant facts in a structured format.

For each document, extract:
1. Pre-impact vehicle actions (turning, stopping, lane change, reversing)
2. Directions (E/W/N/S) and relative vehicle orientations
3. Point of impact on each vehicle (rear-left, front-right, side-swipe, etc.)
4. Maneuvers (U-turn, merging, overtaking)
5. Environmental conditions (weather, lighting, road type, visibility)
6. Traffic conditions (congestion, stationary vehicles)
7. Exact timestamps and location descriptions
8. Implied facts (e.g., "I didn't see him" → visibility obstruction)

For each fact, provide:
- source_text: The exact text from the document
- extracted_fact: A clear statement of the fact
- category: One of: movement, environment, compliance, impact, location, temporal
- confidence: A score from 0.0 to 1.0 indicating how certain you are
- source: The document type (fnol, claimant, other_driver, police, repair_estimate)
- is_implied: true if the fact is implied but not explicitly stated, false otherwise
- normalized_value: A standardized version of the fact value

Return your response as a JSON object with this exact structure:
{
  "facts": [
    {
      "source_text": "string",
      "extracted_fact": "string",
      "category": "movement|environment|compliance|impact|location|temporal",
      "confidence": 0.0-1.0,
      "source": "fnol|claimant|other_driver|police|repair_estimate",
      "is_implied": true|false,
      "normalized_value": "string"
    }
  ]
}

Documents to analyze:
"""
    
    for file_data in files_data:
        filename = file_data.get('filename', file_data.get('originalFilename', 'Unknown'))
        expected_filename = file_data.get('expectedFileName', filename)
        source = identify_document_source(expected_filename)
        file_type = file_data.get('type', 'unknown')
        
        text_content += f"\n--- Document: {filename} (Source: {source}) ---\n"
        
        if file_type == 'pdf':
            pages = file_data.get('pages', [])
            for page in pages:
                page_text = page.get('text', '').strip()
                if page_text:
                    text_content += f"\nPage {page.get('page_number', '?')}:\n{page_text}\n"
            
            # Add images from PDF (with limits)
            for page in pages:
                images = page.get('images', [])
                for img in images:
                    # Limit total images sent to OpenAI API
                    if images_added_count >= MAX_TOTAL_IMAGES_PER_REQUEST:
                        print(f"Warning: Reached image limit ({MAX_TOTAL_IMAGES_PER_REQUEST}), skipping remaining images")
                        break
                    
                    img_data = img.get('data', '')
                    if img_data:
                        if img_data.startswith('data:'):
                            parts = img_data.split(',')
                            if len(parts) == 2:
                                base64_data = parts[1]
                                mime_part = parts[0]
                                if 'image/' in mime_part:
                                    mime_type = mime_part.split('image/')[1].split(';')[0]
                                else:
                                    mime_type = img.get('ext', 'png')
                            else:
                                continue
                        else:
                            base64_data = img_data
                            mime_type = img.get('ext', 'png')
                        
                        # Check individual image size before adding
                        if len(base64_data) > MAX_IMAGE_SIZE_BYTES * 2:  # Allow 2x for base64 overhead
                            print(f"Warning: Skipping large image ({len(base64_data)} bytes) from {filename}")
                            continue
                        
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime_type};base64,{base64_data}"
                            }
                        })
                        images_added_count += 1
                
                if images_added_count >= MAX_TOTAL_IMAGES_PER_REQUEST:
                    break
        
        elif file_type == 'image':
            # Limit total images sent to OpenAI API
            if images_added_count >= MAX_TOTAL_IMAGES_PER_REQUEST:
                print(f"Warning: Reached image limit ({MAX_TOTAL_IMAGES_PER_REQUEST}), skipping image {filename}")
                text_content += f"\nThis is an image file: {filename} (skipped due to image limit)\n"
            else:
                img_data = file_data.get('data', '')
                if img_data:
                    if img_data.startswith('data:'):
                        parts = img_data.split(',')
                        if len(parts) == 2:
                            base64_data = parts[1]
                            mime_part = parts[0]
                            if 'image/' in mime_part:
                                mime_type = mime_part.split('image/')[1].split(';')[0]
                            else:
                                mime_type = file_data.get('format', 'png').lower()
                        else:
                            continue
                    else:
                        base64_data = img_data
                        mime_type = file_data.get('format', 'png').lower()
                    
                    # Check individual image size before adding
                    if len(base64_data) > MAX_IMAGE_SIZE_BYTES * 2:  # Allow 2x for base64 overhead
                        print(f"Warning: Skipping large image ({len(base64_data)} bytes): {filename}")
                        text_content += f"\nThis is an image file: {filename} (skipped due to size limit)\n"
                    else:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime_type};base64,{base64_data}"
                            }
                        })
                        images_added_count += 1
                        text_content += f"\nThis is an image file: {filename}\n"
        
    # Add text content
    content_parts.insert(0, {
        "type": "text",
        "text": text_content
    })
    
    # Call OpenAI API with JSON mode
    try:
        # Estimate content size before API call
        try:
            content_size = sys.getsizeof(str(content_parts))
            if content_size > 20 * 1024 * 1024:  # 20MB warning
                print(f"Warning: Large content payload ({content_size / (1024*1024):.1f}MB) being sent to OpenAI API")
        except Exception:
            pass  # Ignore size estimation errors
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        
        # Clean up content_parts to free memory
        del content_parts
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            facts = result.get('facts', [])
            
            # Create a mapping of document identifiers to sources
            doc_source_map = {}
            doc_text_samples = {}
            for file_data in files_data:
                filename = file_data.get('filename', file_data.get('originalFilename', 'Unknown'))
                expected_filename = file_data.get('expectedFileName', filename)
                source = identify_document_source(expected_filename)
                doc_source_map[expected_filename.lower()] = source
                doc_source_map[filename.lower()] = source
                
                # Store text samples for matching
                if file_data.get('type') == 'pdf':
                    pages = file_data.get('pages', [])
                    text_sample = ' '.join([p.get('text', '')[:200] for p in pages[:2]]).lower()
                    doc_text_samples[source] = text_sample
            
            # Ensure all facts have source properly set
            for fact in facts:
                if not fact.get('source') or fact.get('source') == 'unknown':
                    fact_source_text = fact.get('source_text', '').lower()
                    
                    # Try to match by document name keywords
                    matched = False
                    for doc_name, source in doc_source_map.items():
                        doc_keywords = doc_name.replace('_', ' ').replace('-', ' ').split()
                        if any(keyword in fact_source_text for keyword in doc_keywords if len(keyword) > 3):
                            fact['source'] = source
                            matched = True
                            break
                    
                    # If still not matched, try to match by text content similarity
                    if not matched and fact_source_text:
                        for source, text_sample in doc_text_samples.items():
                            # Check if any significant words from fact appear in document
                            fact_words = set([w for w in fact_source_text.split() if len(w) > 4])
                            doc_words = set([w for w in text_sample.split() if len(w) > 4])
                            if fact_words and len(fact_words.intersection(doc_words)) >= 2:
                                fact['source'] = source
                                break
            
            # Normalize facts
            normalized_facts = normalize_facts(facts)
            
            # Detect conflicts
            conflicts = detect_conflicts(normalized_facts)
            
            return {
                'facts': normalized_facts,
                'conflicts': conflicts
            }
        
        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                facts = result.get('facts', [])
                normalized_facts = normalize_facts(facts)
                conflicts = detect_conflicts(normalized_facts)
                return {
                    'facts': normalized_facts,
                    'conflicts': conflicts
                }
            else:
                raise Exception(f"Failed to parse JSON response: {str(e)}")
    
    except MemoryError as mem_error:
        error_msg = f"Insufficient memory during OpenAI API call. Try reducing the number of images or files."
        print(f"Memory error in extract_facts_from_documents: {mem_error}")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"OpenAI API error: {str(e)}"
        print(f"Error in extract_facts_from_documents: {error_msg}")
        raise Exception(error_msg)


@app.route('/extract-facts', methods=['POST'])
def extract_facts():
    """Extract structured facts from all uploaded files using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get all uploaded files data from request
        if not request.json:
            return jsonify({'error': 'Invalid request: JSON body required'}), 400
        
        files_data = request.json.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided'}), 400
        
        # Memory limit validation: Count total images and check payload size
        total_images = 0
        total_payload_size = 0
        
        try:
            # Estimate payload size and count images
            payload_json = json.dumps(request.json)
            total_payload_size = sys.getsizeof(payload_json)
            
            for file_data in files_data:
                file_type = file_data.get('type', 'unknown')
                
                if file_type == 'pdf':
                    pages = file_data.get('pages', [])
                    for page in pages:
                        images = page.get('images', [])
                        total_images += len(images)
                        
                        # Estimate size of image data
                        for img in images:
                            img_data = img.get('data', '')
                            if img_data:
                                # Base64 data size estimation (rough)
                                total_payload_size += sys.getsizeof(img_data)
                
                elif file_type == 'image':
                    total_images += 1
                    img_data = file_data.get('data', '')
                    if img_data:
                        total_payload_size += sys.getsizeof(img_data)
            
            # Check limits
            if total_images > MAX_TOTAL_IMAGES_PER_REQUEST:
                return jsonify({
                    'error': f'Too many images ({total_images}). Maximum allowed: {MAX_TOTAL_IMAGES_PER_REQUEST}. Please reduce the number of images or split your request.'
                }), 400
            
            # Check payload size (rough estimate: 50MB limit)
            MAX_PAYLOAD_SIZE_MB = 50
            MAX_PAYLOAD_SIZE_BYTES = MAX_PAYLOAD_SIZE_MB * 1024 * 1024
            if total_payload_size > MAX_PAYLOAD_SIZE_BYTES:
                return jsonify({
                    'error': f'Request payload too large (estimated {total_payload_size / (1024*1024):.1f}MB). Maximum allowed: {MAX_PAYLOAD_SIZE_MB}MB. Please reduce file sizes or number of files.'
                }), 400
            
            print(f"Processing request: {total_images} images, ~{total_payload_size / (1024*1024):.1f}MB payload")
        
        except Exception as validation_error:
            print(f"Warning: Error during memory validation: {validation_error}")
            # Continue processing but log the warning
        
        # Extract facts from documents
        try:
            result = extract_facts_from_documents(files_data)
            return jsonify(result), 200
        
        except MemoryError as mem_error:
            error_msg = f'Insufficient memory to process request. Please reduce the number of files or images and try again.'
            print(f"Memory error in fact extraction: {mem_error}")
            return jsonify({'error': error_msg}), 500
        except Exception as e:
            error_msg = str(e)
            print(f"Fact extraction error: {error_msg}")  # Log for debugging
            return jsonify({'error': f'Fact extraction failed: {error_msg}'}), 500
    
    except MemoryError as mem_error:
        error_msg = f'Insufficient memory to process request. Please reduce the number of files or images and try again.'
        print(f"Memory error in request handling: {mem_error}")
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        error_msg = str(e)
        print(f"Request error: {error_msg}")  # Log for debugging
        return jsonify({'error': f'Request failed: {error_msg}'}), 500


@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    """Generate a summary of all uploaded files using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get all uploaded files data from request
        files_data = request.json.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided'}), 400
        
        # Prepare content for OpenAI
        content_parts = []
        
        # Build a text summary of all content first
        text_content = "Please analyze the following auto insurance claim documents and provide a comprehensive summary:\n\n"
        
        for file_data in files_data:
            filename = file_data.get('filename', file_data.get('originalFilename', 'Unknown'))
            file_type = file_data.get('type', 'unknown')
            
            text_content += f"\n--- {filename} ({file_type.upper()}) ---\n"
            
            if file_type == 'pdf':
                # Extract all text from PDF pages
                pages = file_data.get('pages', [])
                for page in pages:
                    page_text = page.get('text', '').strip()
                    if page_text:
                        text_content += f"\nPage {page.get('page_number', '?')}:\n{page_text}\n"
                
                # Add images from PDF as base64 for vision API
                for page in pages:
                    images = page.get('images', [])
                    for img in images:
                        img_data = img.get('data', '')
                        if img_data:
                            # Extract base64 from data URL
                            if img_data.startswith('data:'):
                                parts = img_data.split(',')
                                if len(parts) == 2:
                                    base64_data = parts[1]
                                    mime_part = parts[0]
                                    if 'image/' in mime_part:
                                        mime_type = mime_part.split('image/')[1].split(';')[0]
                                    else:
                                        mime_type = img.get('ext', 'png')
                                else:
                                    continue
                            else:
                                base64_data = img_data
                                mime_type = img.get('ext', 'png')
                            
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mime_type};base64,{base64_data}"
                                }
                            })
            
            elif file_type == 'image':
                # Add image directly
                img_data = file_data.get('data', '')
                if img_data:
                    # Extract base64 from data URL
                    if img_data.startswith('data:'):
                        parts = img_data.split(',')
                        if len(parts) == 2:
                            base64_data = parts[1]
                            mime_part = parts[0]
                            if 'image/' in mime_part:
                                mime_type = mime_part.split('image/')[1].split(';')[0]
                            else:
                                mime_type = file_data.get('format', 'png').lower()
                        else:
                            continue
                    else:
                        base64_data = img_data
                        mime_type = file_data.get('format', 'png').lower()
                    
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{mime_type};base64,{base64_data}"
                        }
                    })
                    
                    # Add description of image
                    text_content += f"\nThis is an image file: {filename}\n"
                    if file_data.get('width') and file_data.get('height'):
                        text_content += f"Image dimensions: {file_data.get('width')}x{file_data.get('height')} pixels\n"
        
        # Add text content
        content_parts.insert(0, {
            "type": "text",
            "text": text_content + "\n\nPlease provide a comprehensive summary of all the claim documents, including:\n1. Key information from each document\n2. Important dates, names, and locations\n3. Damage descriptions\n4. Any inconsistencies or missing information\n5. Overall assessment of the claim"
        })
        
        # Call OpenAI API
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                max_tokens=2000
            )
            
            summary = response.choices[0].message.content
            
            return jsonify({
                'summary': summary,
                'success': True
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Summary generation failed: {str(e)}'}), 500


@app.route('/analyze-liability-signals', methods=['POST'])
def analyze_liability_signals():
    """Analyze fact matrix to identify liability signals using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Build system prompt for liability signal detection
        system_prompt = """You are an expert liability analyst specializing in auto insurance claims. Your task is to identify and analyze liability signals from a fact matrix extracted from claim documents.

Your responsibilities:

1. Identify and label key legal/traffic-control signals:
   - Stop signs, yield signs, traffic lights, uncontrolled intersections
   - Lane position, lane departure, improper lane change
   - Right-of-way determination based on direction + traffic control
   - Speed indicators (explicit or inferred: e.g., "came out of nowhere")
   - Duty-of-care failures (following too closely, distracted driving clues, lack of lookout, unsafe maneuver)
   - Signatures of negligence (e.g., left turn across traffic, reversing into lane)

2. Map signals into a structured Liability Signal Grid with:
   - Signal type: The category of liability signal
   - Evidence text: The supporting text from the fact matrix
   - Impact on liability: Description of how this signal affects liability determination
   - Severity score: A numeric score from 0.0 to 1.0 indicating the severity of the liability impact (0.0 = minimal impact, 1.0 = severe impact)
   - Related facts: Array of fact indices or descriptions that support this signal
   - Discrepancies: Any inconsistencies between signals and claimed facts (not contradictions, just signal inconsistencies)

3. Cross-reference signals with narrative facts (e.g., police notes vs driver statements)

4. Highlight discrepancies between signals and claimed facts (but not as a contradiction module—just signal inconsistency).

Return your response as a JSON object with this exact structure:
{
  "signals": [
    {
      "signal_type": "string (e.g., 'traffic_control', 'right_of_way', 'duty_of_care', 'negligence', 'lane_violation', 'speed_related')",
      "evidence_text": "string - supporting text from facts",
      "impact_on_liability": "string - description of liability impact",
      "severity_score": 0.0-1.0,
      "related_facts": ["array of fact descriptions or indices"],
      "discrepancies": "string - any inconsistencies noted, or empty string if none"
    }
  ]
}

Analyze the following fact matrix:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                signals = result.get('signals', [])
                
                return jsonify({
                    'signals': signals,
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    signals = result.get('signals', [])
                    return jsonify({
                        'signals': signals,
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Liability signals analysis failed: {str(e)}'}), 500


@app.route('/check-evidence-completeness', methods=['POST'])
def check_evidence_completeness():
    """Check completeness of standard evidence package using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Build system prompt for evidence completeness check
        system_prompt = """You are an expert auto insurance claims analyst specializing in evidence package completeness. Your task is to analyze the fact matrix extracted from claim documents and check for completeness of the standard evidence package.

Check for the following:

1. Turn-by-turn incident photos:
   - Are there photos showing the sequence of events?
   - Are photos clear and readable?
   - Do photos show different angles of the incident?

2. Vehicle damage angles:
   - Are both vehicles' damage shown from multiple angles?
   - Are there front, rear, side, and close-up damage photos?
   - Is damage clearly visible and documented?

3. Police report:
   - Is a police report included?
   - Is the police report readable and complete?
   - Does it contain officer ID, narrative, and key details?

4. Timestamps and location data:
   - Are timestamps present on photos or documents?
   - Is location data (address, GPS coordinates) included?
   - Are dates and times consistent across documents?

5. Driver statements completeness:
   - Did either driver omit key factual details?
   - Are statements complete with all required information?
   - Are there gaps in the narrative?

6. Document metadata:
   - Are there missing metadata fields (e.g., missing officer ID, missing narrative section)?
   - Is document metadata complete and consistent?

For each missing or incomplete item, provide:
- evidence_needed: What specific evidence is missing or incomplete
- why_it_matters: Why this evidence is important for the claim
- suggested_follow_up: Suggested questions to ask the claimant to obtain the missing evidence
- priority: "high", "medium", or "low" based on the impact on claim processing

Return your response as a JSON object with this exact structure:
{
  "checks": {
    "turn_by_turn_photos": {
      "present": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    },
    "vehicle_damage_angles": {
      "present": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    },
    "police_report": {
      "present": true|false,
      "readable": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    },
    "timestamps_location": {
      "timestamps_present": true|false,
      "location_data_present": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    },
    "driver_statements": {
      "claimant_complete": true|false,
      "other_driver_complete": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    },
    "document_metadata": {
      "complete": true|false,
      "status": "complete"|"partial"|"missing",
      "details": "string description"
    }
  },
  "missing_evidence": [
    {
      "evidence_needed": "string",
      "why_it_matters": "string",
      "suggested_follow_up": "string",
      "priority": "high"|"medium"|"low"
    }
  ]
}

Analyze the following fact matrix:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                return jsonify({
                    'checks': result.get('checks', {}),
                    'missing_evidence': result.get('missing_evidence', []),
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return jsonify({
                        'checks': result.get('checks', {}),
                        'missing_evidence': result.get('missing_evidence', []),
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Evidence completeness check failed: {str(e)}'}), 500


@app.route('/generate-timeline', methods=['POST'])
def generate_timeline():
    """Generate timeline reconstruction (sequence of events) using OpenAI based on fact matrix."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Build system prompt for timeline reconstruction
        system_prompt = """You are an expert auto insurance claims analyst specializing in timeline reconstruction. Your task is to analyze the fact matrix extracted from claim documents and reconstruct the sequence of events leading to the collision.

Your responsibilities:

1. Analyze all facts from the fact matrix, considering:
   - Temporal facts (timestamps, time of day)
   - Movement facts (vehicle actions, maneuvers)
   - Location facts (positions, directions)
   - Environmental conditions
   - Any sequence indicators in the narrative

2. Reconstruct the timeline as a sequence of events:
   - Event 1: Initial state or first action
   - Event 2: Subsequent action or change
   - Event 3: Next action or change
   - ... (continue as needed)
   - Final Event: The collision/impact

3. For each event, provide:
   - event_number: Sequential number (1, 2, 3, etc.)
   - description: Clear description of what happened in this event
   - timestamp: Time if available from temporal facts, otherwise empty string
   - supporting_facts: Array of fact indices (1-based, e.g., "Fact 1", "Fact 5") that support this event. Each entry should be in the format "Fact {index}" where index is the fact number from the fact matrix (1-based numbering).

4. Ensure logical sequence:
   - Events should flow chronologically
   - Each event should logically lead to the next
   - The final event should be the collision/impact
   - Use temporal facts to order events when available

Return your response as a JSON object with this exact structure:
{
  "timeline": [
    {
      "event_number": 1,
      "description": "string - clear description of the event",
      "timestamp": "string - time if available, otherwise empty string",
      "supporting_facts": ["array of fact indices in format 'Fact 1', 'Fact 5', etc. (1-based numbering from fact matrix)"]
    }
  ]
}

Analyze the following fact matrix:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                timeline = result.get('timeline', [])
                
                return jsonify({
                    'timeline': timeline,
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    timeline = result.get('timeline', [])
                    return jsonify({
                        'timeline': timeline,
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Timeline generation failed: {str(e)}'}), 500


@app.route('/get-liability-recommendation', methods=['POST'])
def get_liability_recommendation():
    """Generate liability percentage recommendation using OpenAI based on fact matrix and liability signals."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix and liability signals data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        signals = request_data.get('signals', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        if not signals:
            return jsonify({'error': 'No liability signals provided. Please analyze liability signals first.'}), 400
        
        # Build system prompt for liability recommendation
        system_prompt = """You are an expert liability analyst specializing in auto insurance claims. Your task is to analyze the fact matrix and liability signals to determine a liability percentage recommendation for the claim.

Your responsibilities:

1. Analyze all facts from the fact matrix, considering:
   - Vehicle movements and actions before impact
   - Traffic control devices and right-of-way rules
   - Environmental conditions
   - Temporal and location facts
   - Any conflicts or contradictions in the facts

2. Evaluate all liability signals, considering:
   - Signal types and their severity scores
   - Impact on liability for each signal
   - Related facts supporting each signal
   - Any discrepancies noted

3. Determine liability split:
   - Calculate percentage of fault for the claimant (0-100%)
   - Calculate percentage of fault for the other driver (0-100%)
   - Ensure percentages sum to 100%
   - Consider comparative negligence principles

4. Provide a detailed explanation:
   - Explain the key factors that influenced the recommendation
   - Reference specific facts and signals that support the percentages
   - Address any conflicts or discrepancies in the evidence
   - Explain the reasoning behind the liability split

5. Identify key factors:
   - List the most important factors that determined the liability split
   - Prioritize factors by their impact on the recommendation

6. Assess confidence:
   - Provide a confidence score (0.0 to 1.0) indicating how certain you are about the recommendation
   - Consider the quality and completeness of evidence

Return your response as a JSON object with this exact structure:
{
  "claimant_liability_percent": 0-100,
  "other_driver_liability_percent": 0-100,
  "explanation": "Detailed explanation of the liability recommendation, including key factors, reasoning, and how facts and signals were weighed in the determination.",
  "key_factors": ["factor1", "factor2", "factor3"],
  "confidence": 0.0-1.0
}

Analyze the following fact matrix and liability signals:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Format signals for the prompt
        signals_text = "\n\nLiability Signals:\n"
        for idx, signal in enumerate(signals):
            signals_text += f"\nSignal {idx + 1}:\n"
            signals_text += f"  Signal Type: {signal.get('signal_type', 'N/A')}\n"
            signals_text += f"  Evidence Text: {signal.get('evidence_text', 'N/A')}\n"
            signals_text += f"  Impact on Liability: {signal.get('impact_on_liability', 'N/A')}\n"
            signals_text += f"  Severity Score: {signal.get('severity_score', 0)}\n"
            signals_text += f"  Related Facts: {', '.join(signal.get('related_facts', []))}\n"
            if signal.get('discrepancies'):
                signals_text += f"  Discrepancies: {signal.get('discrepancies')}\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text + signals_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                # Validate and normalize percentages
                claimant_percent = result.get('claimant_liability_percent', 50)
                other_driver_percent = result.get('other_driver_liability_percent', 50)
                
                # Ensure percentages are integers and sum to 100
                claimant_percent = max(0, min(100, int(round(claimant_percent))))
                other_driver_percent = max(0, min(100, int(round(other_driver_percent))))
                
                # Normalize to sum to 100
                total = claimant_percent + other_driver_percent
                if total != 100 and total > 0:
                    claimant_percent = round((claimant_percent / total) * 100)
                    other_driver_percent = 100 - claimant_percent
                elif total == 0:
                    claimant_percent = 50
                    other_driver_percent = 50
                
                return jsonify({
                    'claimant_liability_percent': claimant_percent,
                    'other_driver_liability_percent': other_driver_percent,
                    'explanation': result.get('explanation', ''),
                    'key_factors': result.get('key_factors', []),
                    'confidence': result.get('confidence', 0.5),
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Validate and normalize percentages
                    claimant_percent = result.get('claimant_liability_percent', 50)
                    other_driver_percent = result.get('other_driver_liability_percent', 50)
                    
                    claimant_percent = max(0, min(100, int(round(claimant_percent))))
                    other_driver_percent = max(0, min(100, int(round(other_driver_percent))))
                    
                    total = claimant_percent + other_driver_percent
                    if total != 100 and total > 0:
                        claimant_percent = round((claimant_percent / total) * 100)
                        other_driver_percent = 100 - claimant_percent
                    elif total == 0:
                        claimant_percent = 50
                        other_driver_percent = 50
                    
                    return jsonify({
                        'claimant_liability_percent': claimant_percent,
                        'other_driver_liability_percent': other_driver_percent,
                        'explanation': result.get('explanation', ''),
                        'key_factors': result.get('key_factors', []),
                        'confidence': result.get('confidence', 0.5),
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Liability recommendation generation failed: {str(e)}'}), 500


@app.route('/generate-claim-rationale', methods=['POST'])
def generate_claim_rationale():
    """Generate audit-ready adjuster narrative rationale using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix and liability signals data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        signals = request_data.get('signals', [])
        files_data = request_data.get('files', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Build system prompt for claim rationale generation
        system_prompt = """You are an expert auto insurance claims adjuster specializing in creating audit-ready claim file rationales. Your task is to generate a clean, structured adjuster-style rationale summarizing the claim.

Your responsibilities:

1. Summarize factual matrix concisely
2. Integrate liability signals into coherent reasoning
3. Explicitly note uncertainty or ambiguous evidence
4. Produce a narrative structured in adjuster format with the following sections:
   - Incident Summary
   - Evidence Overview (narratives + photos)
   - Liability Assessment Logic
   - Key Evidence Supporting Assessment
   - Open Questions / Follow-Up
   - Coverage Considerations
   - Recommendation (not % split)
5. Maintain professional tone suitable for regulatory review
6. Produce minimal fluff; focus on facts + logic + evidence tie-back

Return your response as a JSON object with this exact structure:
{
  "incident_summary": "string - concise summary of the incident",
  "evidence_overview": {
    "narratives": "string - summary of narrative evidence from documents",
    "photos": "string - description of photographic evidence available"
  },
  "liability_assessment_logic": "string - detailed reasoning for liability assessment",
  "key_evidence": ["array of key evidence items supporting the assessment"],
  "open_questions": ["array of open questions or items requiring follow-up"],
  "coverage_considerations": "string - any coverage-related considerations",
  "recommendation": "string - recommendation (not percentage split, but narrative recommendation)"
}

Analyze the following fact matrix and liability signals:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Format signals for the prompt
        signals_text = "\n\nLiability Signals:\n"
        if signals and len(signals) > 0:
            for idx, signal in enumerate(signals):
                signals_text += f"\nSignal {idx + 1}:\n"
                signals_text += f"  Signal Type: {signal.get('signal_type', 'N/A')}\n"
                signals_text += f"  Evidence Text: {signal.get('evidence_text', 'N/A')}\n"
                signals_text += f"  Impact on Liability: {signal.get('impact_on_liability', 'N/A')}\n"
                signals_text += f"  Severity Score: {signal.get('severity_score', 0)}\n"
        else:
            signals_text += "\nNo liability signals provided.\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text + signals_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                # Extract images from uploaded files for display
                images_data = []
                for file_data in files_data:
                    filename = file_data.get('filename', file_data.get('originalFilename', 'Unknown'))
                    expected_filename = file_data.get('expectedFileName', filename)
                    file_type = file_data.get('type', 'unknown')
                    
                    if file_type == 'pdf':
                        pages = file_data.get('pages', [])
                        for page in pages:
                            page_images = page.get('images', [])
                            for img in page_images:
                                images_data.append({
                                    'data': img.get('data', ''),
                                    'source': expected_filename,
                                    'page': page.get('page_number', 0),
                                    'type': 'pdf_image'
                                })
                    elif file_type == 'image':
                        img_data = file_data.get('data', '')
                        if img_data:
                            images_data.append({
                                'data': img_data,
                                'source': expected_filename,
                                'type': 'standalone_image'
                            })
                
                # Add images to response
                result['images'] = images_data
                
                return jsonify({
                    'rationale': result,
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Extract images from uploaded files for display
                    images_data = []
                    for file_data in files_data:
                        filename = file_data.get('filename', file_data.get('originalFilename', 'Unknown'))
                        expected_filename = file_data.get('expectedFileName', filename)
                        file_type = file_data.get('type', 'unknown')
                        
                        if file_type == 'pdf':
                            pages = file_data.get('pages', [])
                            for page in pages:
                                page_images = page.get('images', [])
                                for img in page_images:
                                    images_data.append({
                                        'data': img.get('data', ''),
                                        'source': expected_filename,
                                        'page': page.get('page_number', 0),
                                        'type': 'pdf_image'
                                    })
                        elif file_type == 'image':
                            img_data = file_data.get('data', '')
                            if img_data:
                                images_data.append({
                                    'data': img_data,
                                    'source': expected_filename,
                                    'type': 'standalone_image'
                                })
                    
                    # Add images to response
                    result['images'] = images_data
                    
                    return jsonify({
                        'rationale': result,
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Claim rationale generation failed: {str(e)}'}), 500


@app.route('/generate-escalation-package', methods=['POST'])
def generate_escalation_package():
    """Generate condensed high-risk summary for supervisor escalation using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix and liability signals data from request
        request_data = request.json
        facts = request_data.get('facts', [])
        signals = request_data.get('signals', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Build system prompt for escalation package generation
        system_prompt = """You are an expert auto insurance claims supervisor specializing in identifying high-risk claims requiring escalation. Your task is to generate a manager-ready escalation packet for oversight.

Your responsibilities:

1. Condense claim into high-signal, low-noise document explaining:
   - Why this claim requires supervisor review
   - Key risk drivers
   - Evidence gaps
   - Conflicting statements (if any)
   - Policy-level concerns
2. Output in structured format with:
   - Executive Summary
   - Top 5 Risks
   - Needed Supervisor Decisions
   - Recommended Adjuster Actions
3. Design to reduce supervisor review time from minutes to seconds

Return your response as a JSON object with this exact structure:
{
  "executive_summary": "string - concise executive summary of why this claim requires supervisor review",
  "top_5_risks": [
    {
      "risk": "string - description of the risk",
      "severity": "high|medium|low",
      "impact": "string - description of potential impact"
    }
  ],
  "needed_supervisor_decisions": ["array of decisions that require supervisor input"],
  "recommended_adjuster_actions": ["array of recommended actions for the adjuster"]
}

Analyze the following fact matrix and liability signals:"""
        
        # Format facts for the prompt
        facts_text = "\n\nFact Matrix:\n"
        for idx, fact in enumerate(facts):
            facts_text += f"\nFact {idx + 1}:\n"
            facts_text += f"  Source Text: {fact.get('source_text', 'N/A')}\n"
            facts_text += f"  Extracted Fact: {fact.get('extracted_fact', 'N/A')}\n"
            facts_text += f"  Category: {fact.get('category', 'N/A')}\n"
            facts_text += f"  Source: {fact.get('source', 'N/A')}\n"
            facts_text += f"  Confidence: {fact.get('confidence', 0)}\n"
            if fact.get('normalized_value'):
                facts_text += f"  Normalized Value: {fact.get('normalized_value')}\n"
        
        # Format signals for the prompt
        signals_text = "\n\nLiability Signals:\n"
        if signals and len(signals) > 0:
            for idx, signal in enumerate(signals):
                signals_text += f"\nSignal {idx + 1}:\n"
                signals_text += f"  Signal Type: {signal.get('signal_type', 'N/A')}\n"
                signals_text += f"  Evidence Text: {signal.get('evidence_text', 'N/A')}\n"
                signals_text += f"  Impact on Liability: {signal.get('impact_on_liability', 'N/A')}\n"
                signals_text += f"  Severity Score: {signal.get('severity_score', 0)}\n"
        else:
            signals_text += "\nNo liability signals provided.\n"
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text + signals_text
        
        # Call OpenAI API with JSON mode
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                return jsonify({
                    'escalation_package': result,
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return jsonify({
                        'escalation_package': result,
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Escalation package generation failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)

