import os
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure uploads directory exists
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
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text() or ''
            
            # Extract images from pdfplumber
            images = []
            if page.images:
                for img_index, img_obj in enumerate(page.images):
                    try:
                        # Get image bounding box
                        bbox = (img_obj.get('x0', 0), img_obj.get('top', 0), 
                                img_obj.get('x1', 0), img_obj.get('bottom', 0))
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            cropped = page.crop(bbox)
                            
                            # Try to get image as PIL Image using pdfplumber's to_image
                            try:
                                pil_image = cropped.to_image(resolution=150)
                                img_buffer = BytesIO()
                                pil_image.save(img_buffer, format='PNG')
                                img_bytes = img_buffer.getvalue()
                                
                                # Convert to base64
                                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                                images.append({
                                    'index': img_index,
                                    'data': f"data:image/png;base64,{image_base64}",
                                    'ext': 'png'
                                })
                            except Exception as img_error:
                                print(f"Error converting image {img_index} from page {page_num} to PIL: {img_error}")
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
    
    # Also try to extract images using PyPDF2 as fallback
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    if '/Resources' in page and '/XObject' in page['/Resources']:
                        xobjects = page['/Resources']['/XObject']
                        if hasattr(xobjects, 'get_object'):
                            xobjects = xobjects.get_object()
                        
                        img_index_offset = len(result['pages'][page_num]['images'])
                        for obj_name, obj in xobjects.items():
                            if hasattr(obj, 'get') and obj.get('/Subtype') == '/Image':
                                try:
                                    # Extract image data
                                    data = obj.get_data()
                                    
                                    # Determine image format
                                    if '/Filter' in obj:
                                        filter_type = obj['/Filter']
                                        if isinstance(filter_type, list):
                                            filter_type = filter_type[0] if filter_type else ''
                                        if filter_type == '/DCTDecode':
                                            ext = 'jpg'
                                        elif filter_type == '/FlateDecode':
                                            ext = 'png'
                                        else:
                                            ext = 'png'
                                    else:
                                        ext = 'png'
                                    
                                    # Convert to base64
                                    image_base64 = base64.b64encode(data).decode('utf-8')
                                    
                                    # Add to images
                                    if page_num < len(result['pages']):
                                        new_index = len(result['pages'][page_num]['images'])
                                        result['pages'][page_num]['images'].append({
                                            'index': new_index,
                                            'data': f"data:image/{ext};base64,{image_base64}",
                                            'ext': ext
                                        })
                                except Exception as e:
                                    print(f"Error extracting image with PyPDF2 from page {page_num}: {e}")
                except Exception as e:
                    print(f"Error processing page {page_num} with PyPDF2: {e}")
    except Exception as e:
        print(f"Error in PyPDF2 image extraction: {e}")
    
    return result


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


def extract_image_content(image_path):
    """
    Extract content from an image file.
    Returns a dictionary with image data.
    """
    try:
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            
            # Get image format
            format_name = img.format or 'PNG'
            
            # Convert to base64
            img_buffer = BytesIO()
            # Save in original format if possible, otherwise PNG
            save_format = format_name if format_name in ['JPEG', 'PNG', 'GIF', 'WEBP'] else 'PNG'
            img.save(img_buffer, format=save_format)
            img_bytes = img_buffer.getvalue()
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Determine MIME type
            mime_type_map = {
                'JPEG': 'image/jpeg',
                'PNG': 'image/png',
                'GIF': 'image/gif',
                'WEBP': 'image/webp'
            }
            mime_type = mime_type_map.get(save_format, 'image/png')
            
            return {
                'type': 'image',
                'filename': os.path.basename(image_path),
                'width': width,
                'height': height,
                'format': save_format,
                'data': f"data:{mime_type};base64,{image_base64}",
                'size': len(img_bytes)
            }
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
            
            # Add images from PDF
            for page in pages:
                images = page.get('images', [])
                for img in images:
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
                        
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime_type};base64,{base64_data}"
                            }
                        })
        
        elif file_type == 'image':
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
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime_type};base64,{base64_data}"
                    }
                })
                
                text_content += f"\nThis is an image file: {filename}\n"
        
    # Add text content
    content_parts.insert(0, {
        "type": "text",
        "text": text_content
    })
    
    # Call OpenAI API with JSON mode
    try:
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
    
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


@app.route('/extract-facts', methods=['POST'])
def extract_facts():
    """Extract structured facts from all uploaded files using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get all uploaded files data from request
        files_data = request.json.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided'}), 400
        
        # Extract facts from documents
        try:
            result = extract_facts_from_documents(files_data)
            return jsonify(result), 200
        
        except Exception as e:
            return jsonify({'error': f'Fact extraction failed: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Request failed: {str(e)}'}), 500


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


@app.route('/analyze-coverage-impact', methods=['POST'])
def analyze_coverage_impact():
    """Analyze fact matrix and files to identify coverage impact signals using OpenAI."""
    try:
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.'}), 500
        
        # Get fact matrix data and files from request
        request_data = request.json
        facts = request_data.get('facts', [])
        files_data = request_data.get('files', [])
        
        if not facts:
            return jsonify({'error': 'No facts provided. Please extract facts first.'}), 400
        
        # Find policy document if uploaded
        policy_document = None
        for file_data in files_data:
            filename = file_data.get('filename', file_data.get('originalFilename', ''))
            expected_filename = file_data.get('expectedFileName', filename)
            source = identify_document_source(expected_filename)
            if source == 'policy':
                policy_document = file_data
                break
        
        # Build system prompt for coverage impact detection
        system_prompt = """You are an expert auto insurance coverage analyst specializing in identifying facts that may affect coverage decisions or create exceptions, even if they don't affect liability.

Your responsibilities:

1. Identify any statements implying coverage exclusions:
   - Commercial use of personal vehicle (e.g., delivery, ride-sharing, business use)
   - Unlisted driver involved (driver not on policy)
   - Vehicle not primarily garaged at declared address (garaging violations)
   - Time-of-loss inconsistencies (timing issues that suggest fraud or misrepresentation)
   - Suspicious locations (e.g., accident far from home state, unusual locations)
   - Other coverage exclusions based on policy terms

2. Detect mismatches between FNOL and repair estimate:
   - Prior damage mentioned in repair estimate but not in FNOL
   - Damage descriptions that don't align
   - Cost discrepancies that suggest pre-existing conditions
   - Timeline inconsistencies between documents

3. Cross-reference with policy document (if provided):
   - Extract relevant coverage clauses and exclusions
   - Match triggering facts to specific policy provisions
   - Identify policy violations or exceptions

4. Produce Coverage Impact Summary with:
   - triggering_fact: The specific fact that triggers a coverage concern
   - source: The document source where this fact was found (fnol, claimant, other_driver, police, repair_estimate, policy)
   - policy_clause: The relevant policy clause or exclusion if policy document is available, otherwise "Policy document not provided"
   - risk_level: "high", "medium", or "low" based on the severity of coverage impact
   - coverage_exclusion_type: Type of coverage exclusion (e.g., "commercial_use", "unlisted_driver", "garaging_violation", "time_inconsistency", "location_suspicious", "prior_damage_mismatch", "other")
   - explanation: A clear explanation of why this fact affects coverage
   - evidence: Supporting evidence text from the documents

Return your response as a JSON object with this exact structure:
{
  "coverage_impacts": [
    {
      "triggering_fact": "string",
      "source": "string",
      "policy_clause": "string",
      "risk_level": "high|medium|low",
      "coverage_exclusion_type": "string",
      "explanation": "string",
      "evidence": "string"
    }
  ]
}

Analyze the following fact matrix and documents:"""
        
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
        
        # Add policy document content if available
        policy_text = ""
        content_parts = []
        
        if policy_document:
            policy_text += "\n\n--- Policy Document ---\n"
            file_type = policy_document.get('type', 'unknown')
            
            if file_type == 'pdf':
                pages = policy_document.get('pages', [])
                for page in pages:
                    page_text = page.get('text', '').strip()
                    if page_text:
                        policy_text += f"\nPage {page.get('page_number', '?')}:\n{page_text}\n"
                
                # Add images from policy PDF
                for page in pages:
                    images = page.get('images', [])
                    for img in images:
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
                            
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mime_type};base64,{base64_data}"
                                }
                            })
        
        # Prepare content for OpenAI
        full_prompt = system_prompt + facts_text + policy_text
        
        # Add text content first
        content_parts.insert(0, {
            "type": "text",
            "text": full_prompt
        })
        
        # Call OpenAI API with JSON mode
        try:
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
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                coverage_impacts = result.get('coverage_impacts', [])
                
                return jsonify({
                    'coverage_impacts': coverage_impacts,
                    'success': True
                }), 200
            
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    coverage_impacts = result.get('coverage_impacts', [])
                    return jsonify({
                        'coverage_impacts': coverage_impacts,
                        'success': True
                    }), 200
                else:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
        
        except Exception as e:
            return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Coverage impact analysis failed: {str(e)}'}), 500


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

