from flask import Flask, request, jsonify, render_template
import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer
from pinecone import Pinecone
import fitz  # PyMuPDF
import io
from PIL import Image
import numpy as np
import uuid
import logging
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PDFExtractor:
    def __init__(self):
        # Base directory for all educational content
        self.base_dir = r"C:\Users\Ajithkumar.p\Downloads\books"
        
        # Define grade structure with folders
        self.grade_structure = {
            "8": {
                "folder": os.path.join(self.base_dir, "8th_standard"),
                "name": "8th Standard"
            },
            "9": {
                "folder": os.path.join(self.base_dir, "9th_standard"),
                "name": "9th Standard"
            },
            "10": {
                "folder": os.path.join(self.base_dir, "10th_standard"),
                "name": "10th Standard"
            },
            "11": {
                "folder": os.path.join(self.base_dir, "11th_standard"),
                "name": "11th Standard"
            },
            "12": {
                "folder": os.path.join(self.base_dir, "12th_standard"),
                "name": "12th Standard"
            }
        }
        
        # Create grade folders if they don't exist
        for grade in self.grade_structure.values():
            os.makedirs(grade["folder"], exist_ok=True)
        
        # Automatically discover PDF files in each grade folder
        self.pdf_files = self.discover_pdf_files()
        
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.pc = None
        self.index_name = "education-pdf-index"
        
        # Map of grade levels - will be used to tailor responses
        self.grade_levels = {
            "8": "8th Standard",
            "9": "9th Standard",
            "10": "10th Standard",
            "11": "11th Standard",
            "12": "12th Standard"
        }
        
        # Subject categorization
        self.subjects = {
            "math": ["mathematics", "algebra", "geometry", "calculus", "arithmetic", "trigonometry"],
            "science": ["physics", "chemistry", "biology", "science", "experiment"],
            "social": ["history", "geography", "civics", "economics", "social studies"],
            "literature": ["tamil", "english", "literature", "language", "grammar", "poem", "story"]
        }
        
        # Track which grade level is currently active for chat
        self.active_grade = None
        
        # Initialize models and Pinecone
        self.initialize_model()
        self.initialize_pinecone()
    
    def discover_pdf_files(self):
        """Automatically discover PDF files in each grade folder"""
        pdf_files = []
        
        for grade_key, grade_info in self.grade_structure.items():
            folder = grade_info["folder"]
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith('.pdf'):
                        full_path = os.path.join(folder, filename)
                        pdf_files.append(full_path)
                        logger.info(f"Discovered PDF: {full_path} for grade {grade_key}")
        
        return pdf_files
    
    def initialize_model(self):
        """Initialize the CLIP model, tokenizer and processor"""
        try:
            logger.info("Loading CLIP model...")
            # Using OpenAI's CLIP model which is publicly available
            model_name = "openai/clip-vit-base-patch32"
            
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"CLIP model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error initializing CLIP model: {str(e)}")
            raise
    
    def initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            logger.info("Connecting to Pinecone...")
            # Initialize Pinecone with API key from environment variables
            api_key = os.getenv("PINECONE_API_KEY")
            
            if not api_key:
                logger.error("PINECONE_API_KEY environment variable not set")
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=api_key)
            
            # Check if index exists, if not create it
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                # Create the index - CLIP has 512 dimensions
                self.pc.create_index(
                    name=self.index_name,
                    dimension=512,  # CLIP embedding dimension
                    metric="cosine"
                )
            
            # Get a reference to the index
            self.index = self.pc.Index(self.index_name)
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def determine_grade_and_subject(self, pdf_path):
        """Determine grade level and subject based on file path and name"""
        filename = os.path.basename(pdf_path).lower()
        filepath = os.path.dirname(pdf_path).lower()
        
        # Determine grade level from the folder structure
        grade_level = None
        for grade_key, grade_info in self.grade_structure.items():
            if grade_info["folder"].lower() in filepath:
                grade_level = grade_info["name"]
                break
                
        # If not found in path, try from filename
        if not grade_level:
            for grade in self.grade_levels:
                if grade in filename or f"standard{grade}" in filename or f"std{grade}" in filename:
                    grade_level = self.grade_levels[grade]
                    break
        
        # Determine subject
        subject = None
        for subj, keywords in self.subjects.items():
            if any(keyword in filename for keyword in keywords):
                subject = subj
                break
                
        return grade_level, subject
    
    def extract_text_and_images_from_pdf(self, pdf_path):
        """Extract text and images from PDF files with enhanced metadata"""
        try:
            logger.info(f"Extracting content from: {pdf_path}")
            document = fitz.open(pdf_path)
            content_blocks = []
            
            # Determine grade level and subject from file path and name
            grade_level, subject = self.determine_grade_and_subject(pdf_path)
            
            for page_num, page in enumerate(document):
                # Extract text - break into meaningful chunks by paragraphs
                text = page.get_text()
                
                # Process text into meaningful chunks
                paragraphs = re.split(r'\n\s*\n', text)
                for para_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        # Check if it's a heading (shorter text, often with numbers)
                        is_heading = len(paragraph.strip()) < 100 and (
                            re.match(r'^[0-9]+\.', paragraph.strip()) or 
                            paragraph.strip().isupper() or
                            re.match(r'^[A-Z][\w\s]+:$', paragraph.strip())
                        )
                        
                        content_blocks.append({
                            "type": "text",
                            "content": paragraph.strip(),
                            "page": page_num + 1,
                            "paragraph_index": para_idx,
                            "is_heading": is_heading,
                            "grade_level": grade_level,
                            "subject": subject
                        })
                
                # Extract images with captions (text near images)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image location on the page
                    rect = page.get_image_bbox(xref)
                    
                    # Try to find a caption (text within 100 points below the image)
                    caption_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 100)
                    caption_text = page.get_text("text", clip=caption_rect).strip()
                    
                    # Convert image bytes to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    content_blocks.append({
                        "type": "image",
                        "content": image,
                        "page": page_num + 1,
                        "image_index": img_index,
                        "caption": caption_text,
                        "grade_level": grade_level,
                        "subject": subject
                    })
            
            logger.info(f"Extracted {len(content_blocks)} content blocks from {pdf_path}")
            return content_blocks
        except Exception as e:
            logger.error(f"Error extracting content from PDF {pdf_path}: {str(e)}")
            return []
    
    def generate_embeddings(self, content_block):
        """Generate embeddings for text or image content using CLIP model"""
        try:
            if content_block["type"] == "text":
                # Process text with CLIP
                inputs = self.tokenizer(
                    content_block["content"], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=77  # CLIP's max token length
                ).to(self.device)
                
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                
                # Normalize the features
                embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
                
            elif content_block["type"] == "image":
                # Process image with CLIP
                image = content_block["content"]
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize the features
                embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
            
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def index_pdf_files(self, grade_level=None):
        """Process and index all PDF files, optionally filtering by grade level"""
        total_indexed = 0
        
        # Filter PDF files by grade level if specified
        files_to_process = []
        if grade_level:
            grade_folder = None
            for grade_key, grade_info in self.grade_structure.items():
                if grade_info["name"] == grade_level:
                    grade_folder = grade_info["folder"].lower()
                    break
            
            if grade_folder:
                files_to_process = [pdf for pdf in self.pdf_files if grade_folder in pdf.lower()]
        else:
            files_to_process = self.pdf_files
        
        for pdf_file in files_to_process:
            try:
                if not os.path.exists(pdf_file):
                    logger.error(f"PDF file not found: {pdf_file}")
                    continue
                
                logger.info(f"Processing file: {pdf_file}")
                filename = os.path.basename(pdf_file)
                
                # Extract content blocks from PDF
                content_blocks = self.extract_text_and_images_from_pdf(pdf_file)
                
                # Process and index each content block
                vectors_to_upsert = []
                
                for block in content_blocks:
                    # Generate a unique ID for each content block
                    block_id = str(uuid.uuid4())
                    
                    # Generate embeddings
                    if block["type"] == "text":
                        embedding = self.generate_embeddings(block)
                        
                        if embedding is not None:
                            metadata = {
                                "filename": filename,
                                "page": block["page"],
                                "type": "text",
                                "content": block["content"][:1000],  # Store shortened content for reference
                                "is_heading": block.get("is_heading", False),
                                "grade_level": block.get("grade_level"),
                                "subject": block.get("subject")
                            }
                            
                            vectors_to_upsert.append({
                                "id": block_id,
                                "values": embedding.tolist(),
                                "metadata": metadata
                            })
                    
                    elif block["type"] == "image":
                        # For images, we'll store a reference and caption
                        embedding = self.generate_embeddings(block)
                        
                        if embedding is not None:
                            metadata = {
                                "filename": filename,
                                "page": block["page"],
                                "type": "image",
                                "image_index": block["image_index"],
                                "caption": block.get("caption", ""),
                                "grade_level": block.get("grade_level"),
                                "subject": block.get("subject")
                            }
                            
                            vectors_to_upsert.append({
                                "id": block_id,
                                "values": embedding.tolist(),
                                "metadata": metadata
                            })
                
                # Batch upsert to Pinecone (in chunks of 100 to avoid hitting limits)
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                total_indexed += len(vectors_to_upsert)
                logger.info(f"Indexed {len(vectors_to_upsert)} vectors from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        return total_indexed
    
    def search(self, query, grade_level=None, top_k=8):
        """Search the vector database using the query"""
        try:
            # Generate embedding for the query text
            query_block = {
                "type": "text",
                "content": query
            }
            query_embedding = self.generate_embeddings(query_block)
            
            if query_embedding is None:
                return {"error": "Failed to generate embedding for query"}
            
            # Prepare filter if grade level is specified
            filter_dict = {}
            if grade_level:
                filter_dict["grade_level"] = {"$eq": grade_level}
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "score": match.score,
                    "filename": match.metadata.get("filename"),
                    "page": match.metadata.get("page"),
                    "type": match.metadata.get("type"),
                    "grade_level": match.metadata.get("grade_level"),
                    "subject": match.metadata.get("subject")
                }
                
                if match.metadata.get("type") == "text":
                    result["content"] = match.metadata.get("content")
                    result["is_heading"] = match.metadata.get("is_heading", False)
                elif match.metadata.get("type") == "image":
                    result["caption"] = match.metadata.get("caption", "")
                
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return {"error": str(e)}
    
    def format_educational_response(self, results, query, grade_level=None):
        """Format search results into an educational response suitable for students"""
        if not results or isinstance(results, dict) and "error" in results:
            return {
                "answer": "I'm sorry, I couldn't find information about that in my knowledge base.",
                "sources": []
            }
        
        # Extract the most relevant content
        text_results = [r for r in results if r["type"] == "text"]
        
        # Start with headings if available
        headings = [r for r in text_results if r.get("is_heading", False)]
        content_items = [r for r in text_results if not r.get("is_heading", False)]
        
        # Compile sources for citation
        sources = []
        for result in results[:3]:  # Top 3 sources
            if result["type"] == "text":
                sources.append({
                    "book": result["filename"],
                    "page": result["page"],
                    "excerpt": result["content"][:100] + "..."
                })
        
        # Compile main content for the answer
        main_content = []
        
        # Add relevant headings first
        for heading in headings[:2]:
            main_content.append(heading["content"])
            
        # Add content paragraphs
        for item in content_items[:3]:
            main_content.append(item["content"])
        
        # Generate a simplified answer based on grade level
        if grade_level and grade_level in ["8th Standard", "9th Standard"]:
            # Simplify language for younger students
            combined_text = " ".join(main_content)
            # Break into shorter sentences
            answer = re.sub(r'(\. )', '.\n', combined_text)
            # Limit to first 500 characters for younger students
            answer = answer[:500] + "..." if len(answer) > 500 else answer
        else:
            # For older students, provide more comprehensive information
            combined_text = " ".join(main_content)
            answer = combined_text
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def set_active_grade(self, grade_level):
        """Set the active grade level for subsequent chat interactions"""
        self.active_grade = grade_level
        logger.info(f"Active grade level set to: {grade_level}")
        return {"status": "success", "message": f"Active grade level set to {grade_level}"}
    
    def get_chat_response(self, query, grade_level=None):
        """Process a student query and return an educational response"""
        # Use active grade if set and no specific grade provided
        if not grade_level and self.active_grade:
            grade_level = self.active_grade
            logger.info(f"Using active grade level: {grade_level}")
        
        # Search for relevant content
        results = self.search(query, grade_level=grade_level)
        
        # Format the response for educational purposes
        formatted_response = self.format_educational_response(results, query, grade_level)
        
        # Add grade-specific introduction if applicable
        if grade_level:
            grade_intro = f"Based on the {grade_level} curriculum: "
            formatted_response["answer"] = grade_intro + formatted_response["answer"]
        
        # Add related questions based on search results
        related_questions = self.generate_related_questions(results, query, grade_level)
        formatted_response["related_questions"] = related_questions
        
        return formatted_response
    
    def generate_related_questions(self, results, original_query, grade_level=None):
        """Generate related questions based on search results and grade level"""
        related_questions = []
        
        # Extract key topics from content
        topics = set()
        for result in results:
            if result["type"] == "text":
                content = result["content"].lower()
                
                # Extract potential topics (nouns and noun phrases)
                words = content.split()
                for word in words:
                    # Simple filtering for potential topic words (at least 4 chars, not in original query)
                    if len(word) > 3 and word not in original_query.lower():
                        topics.add(word)
        
        # Generate grade-appropriate related questions
        if grade_level:
            if grade_level in ["8th Standard", "9th Standard"]:
                # Simpler questions for younger students
                if topics:
                    topic = list(topics)[0] if topics else "this topic"
                    related_questions.append(f"What is {topic} used for?")
                    if len(topics) > 1:
                        topic2 = list(topics)[1]
                        related_questions.append(f"Can you give an example of {topic2}?")
            else:
                # More complex questions for older students
                if topics:
                    topic = list(topics)[0] if topics else "this concept"
                    related_questions.append(f"How does {topic} relate to real-world applications?")
                    if len(topics) > 1:
                        topic2 = list(topics)[1]
                        related_questions.append(f"What are the theoretical foundations of {topic2}?")
        
        # Add general questions if we don't have enough
        if len(related_questions) < 2:
            if "what" in original_query.lower():
                related_questions.append("How can this be applied in practical situations?")
            if "how" in original_query.lower():
                related_questions.append("What are the key principles behind this?")
            
            # If still not enough, add a default
            if len(related_questions) < 2:
                related_questions.append("Could you explain this in more detail?")
        
        return related_questions[:3]  # Return top 3 related questions

# Initialize the extractor
extractor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index-pdfs', methods=['POST'])
def index_pdfs():
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        # Get optional grade level filter from request
        data = request.get_json() or {}
        grade_level = data.get('grade_level')
        
        total_indexed = extractor.index_pdf_files(grade_level=grade_level)
        
        message = f"Indexed {total_indexed} content blocks"
        if grade_level:
            message += f" for {grade_level}"
        
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        logger.error(f"Error in index-pdfs endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set-grade', methods=['POST'])
def set_grade():
    """Set the active grade level for subsequent chat interactions"""
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        data = request.get_json()
        grade_level = data.get('grade_level')
        
        if not grade_level:
            return jsonify({"status": "error", "message": "Grade level is required"}), 400
        
        result = extractor.set_active_grade(grade_level)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in set-grade endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        data = request.get_json()
        query = data.get('query')
        grade_level = data.get('grade_level')  # Optional grade level filter
        
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400
        
        results = extractor.search(query, grade_level=grade_level)
        
        # Format results in an educational way
        formatted_response = extractor.format_educational_response(results, query, grade_level)
        
        return jsonify({
            "status": "success", 
            "formatted_response": formatted_response,
            "raw_results": results
        })
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint for the study assistant chatbot"""
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        data = request.get_json()
        query = data.get('message')
        grade_level = data.get('grade_level')  # Optional grade level
        
        if not query:
            return jsonify({"status": "error", "message": "Message is required"}), 400
        
        # Get chat response
        response = extractor.get_chat_response(query, grade_level=grade_level)
        
        return jsonify({
            "status": "success", 
            "response": response
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get-subjects', methods=['GET'])
def get_subjects():
    """Return available subjects from indexed content"""
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        return jsonify({
            "status": "success", 
            "subjects": list(extractor.subjects.keys())
        })
    except Exception as e:
        logger.error(f"Error in get-subjects endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get-grade-levels', methods=['GET'])
def get_grade_levels():
    """Return available grade levels from indexed content"""
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        return jsonify({
            "status": "success", 
            "grade_levels": list(extractor.grade_levels.values())
        })
    except Exception as e:
        logger.error(f"Error in get-grade-levels endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/list-books', methods=['GET'])
def list_books():
    """List available books by grade level"""
    global extractor
    
    try:
        if extractor is None:
            extractor = PDFExtractor()
        
        grade_level = request.args.get('grade_level')
        
        books_by_grade = {}
        
        for grade_key, grade_info in extractor.grade_structure.items():
            grade_name = grade_info["name"]
            folder = grade_info["folder"]
            
            if not grade_level or grade_name == grade_level:
                books = []
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        if filename.lower().endswith('.pdf'):
                            # Try to determine subject
                            subject = None
                            for subj, keywords in extractor.subjects.items():
                                if any(keyword in filename.lower() for keyword in keywords):
                                    subject = subj
                                    break
                            
                            books.append({
                                "filename": filename,
                                "subject": subject
                            })
                
                books_by_grade[grade_name] = books
        
        return jsonify({
            "status": "success",
            "books": books_by_grade
        })
    except Exception as e:
        logger.error(f"Error in list-books endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
