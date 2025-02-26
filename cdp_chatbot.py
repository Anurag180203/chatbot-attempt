from typing import Dict, List, Optional
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

class CDPChatbot:
    def __init__(self):
        # Add OpenAI configuration
        self.openai_client = OpenAI()  # Requires api_key setup
        
        # Initialize NLTK components
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentence transformer model for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # CDP documentation URLs
        self.cdp_urls = {
            'segment': 'https://segment.com/docs/',
            'mparticle': 'https://docs.mparticle.com/',
            'lytics': 'https://docs.lytics.com/',
            'zeotap': 'https://docs.zeotap.com/'
        }
        
        # Store processed documentation
        self.documentation = defaultdict(list)
        self.doc_embeddings = {}
        
        # Initialize documentation
        self._fetch_documentation()

    def _fetch_documentation(self):
        """Fetch and process documentation from CDP websites."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        for cdp, base_url in self.cdp_urls.items():
            print(f"Fetching documentation for {cdp}...")
            try:
                # Fetch main documentation page
                response = requests.get(base_url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                soup = BeautifulSoup(response.text, 'html.parser')
                
                print(f"Processing main page for {cdp}...")
                # Extract text content and links
                self._process_page(cdp, soup, base_url)
                
                # Process internal documentation links
                links = self._get_documentation_links(soup, base_url)
                total_links = min(len(links), 10)  # Limit to first 10 links
                
                print(f"Found {total_links} subpages to process for {cdp}")
                for i, link in enumerate(links[:10], 1):
                    try:
                        print(f"Processing subpage {i}/{total_links} for {cdp}: {link}")
                        time.sleep(5)  # Rate limiting
                        response = requests.get(link, headers=headers, timeout=10)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, 'html.parser')
                        self._process_page(cdp, soup, base_url)
                    except requests.RequestException as e:
                        print(f"Warning: Failed to process link {link}: {str(e)}")
                        continue
                
                print(f"Generating embeddings for {cdp} documentation...")
                self._generate_embeddings(cdp)
                print(f"Successfully processed {cdp} documentation")
                
            except requests.RequestException as e:
                print(f"Error: Failed to fetch documentation for {cdp}: {str(e)}")
            except Exception as e:
                print(f"Error: Unexpected error processing {cdp}: {str(e)}")

    def _process_page(self, cdp: str, soup: BeautifulSoup, base_url: str):
        """Extract and process relevant content from a documentation page with improved handling."""
        try:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text from main content area with CDP-specific selectors
            content_selectors = {
                'segment': ['.docs-content', '.markdown', '.markdown-body', '.documentation'],
                'mparticle': ['.content', '.doc-content', '.main-content', '.markdown', '.markdown-body'],
                'lytics': ['.documentation-content', '.content-body', '.docs-main', '.markdown', '.markdown-body'],
                'zeotap': ['.main-content', '.documentation', '.content', '.markdown', '.markdown-body']
            }

            content = None
            for selector in content_selectors.get(cdp, ['main']):
                content = soup.select_one(selector)
                if content:
                    break

            if not content:
                print(f"Warning: Could not find main content area for {base_url}")
                return

            # Extract text and clean it
            text = content.get_text(separator=' ', strip=True)
            
            # Split into meaningful chunks
            chunks = self._split_into_chunks(text)
            
            # Store relevant chunks
            relevant_chunks = 0
            for chunk in chunks:
                if self._is_relevant_chunk(chunk):
                    processed_chunk = self._process_chunk(chunk)
                    self.documentation[cdp].append({
                        'text': processed_chunk,
                        'url': base_url
                    })
                    relevant_chunks += 1
            
            print(f"Extracted {relevant_chunks} relevant chunks from {base_url}")

        except Exception as e:
            print(f"Error: Failed to process page {base_url}: {str(e)}")

    def _get_documentation_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant documentation links from the page."""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            if base_url in full_url and '#' not in href:
                links.append(full_url)
        return list(set(links))

    def _generate_embeddings(self, cdp: str):
        """Generate embeddings for the documentation content."""
        texts = [doc['text'] for doc in self.documentation[cdp]]
        if texts:
            self.doc_embeddings[cdp] = self.model.encode(texts)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess the input text by tokenizing, removing stopwords, and lemmatizing."""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return tokens

    def _identify_cdp(self, question: str) -> Optional[str]:
        """
        Identify which CDP the question is about, with improved handling of variations.
        Returns None if no CDP is mentioned or if question is irrelevant.
        """
        question = question.lower()
        
        # Check if question is CDP-related
        cdp_related_terms = [
            'cdp', 'platform', 'data', 'analytics', 'track', 'segment', 'profile',
            'audience', 'integration', 'source', 'destination', 'event', 'mparticle', 'lytics', 'zeotap'
        ]
        
        # If the question is too long, focus on the main part
        if len(question.split()) > 50:
            # Take first and last 25 words
            words = question.split()
            question = ' '.join(words[:25] + words[-25:])
        
        # Check if question is CDP-related
        if not any(term in question for term in cdp_related_terms):
            return None
        
        # Check for specific CDP mentions
        for cdp in self.cdp_urls.keys():
            if cdp in question:
                return cdp
            
        return None

    def _identify_intent(self, tokens: List[str]) -> str:
        """Identify the intent of the question based on preprocessed tokens."""
        intent_keywords = {
            'track_event': ['track', 'event', 'log', 'record'],
            'identify_user': ['identify', 'user', 'profile'],
            'integration': ['integrate', 'connection', 'connect'],
            # Add more intents and their keywords
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in tokens for keyword in keywords):
                return intent
        return None

    def answer_question(self, question: str) -> str:
        """
        Enhanced answer generation using OpenAI with documentation context.
        """
        try:
            cdp = self._identify_cdp(question)
            if cdp is None:
                return "Please specify which CDP you're asking about (Segment, mParticle, Lytics, or Zeotap)."

            if cdp not in self.doc_embeddings or len(self.doc_embeddings[cdp]) == 0:
                return f"Error: Documentation for {cdp} has not been properly loaded."

            # Get relevant documentation using existing semantic search
            cleaned_question = self._clean_question(question)
            question_embedding = self.model.encode(cleaned_question)
            
            relevant_docs = []
            similarities = np.dot(self.doc_embeddings[cdp], question_embedding)
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0.3:
                    relevant_docs.append(self.documentation[cdp][idx]['text'])

            if not relevant_docs:
                return f"I couldn't find specific information about that in the {cdp} documentation. Please try rephrasing your question."

            # Construct prompt for OpenAI
            system_prompt = f"""You are a helpful CDP documentation assistant for {cdp}. 
            Use the provided documentation to answer the user's question accurately and concisely.
            If the documentation doesn't contain enough information to answer fully, acknowledge that."""

            context = "\n\n".join(relevant_docs)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Documentation context:\n{context}\n\nQuestion: {question}"}
            ]

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                source_url = self.documentation[cdp][top_indices[0]]['url']
                
                return f"{answer}\n\nSource: {source_url}"
                
            except Exception as e:
                # Fallback to existing response generation
                return self._generate_basic_response(relevant_docs)

        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def _generate_basic_response(self, relevant_docs: List[str]) -> str:
        """Fallback method for when OpenAI is unavailable."""
        response = "Here's what I found:\n\n"
        response += "\n\n".join(relevant_docs)
        return response

    def _clean_question(self, question: str) -> str:
        """Clean and normalize the question text."""
        # Remove multiple spaces and normalize whitespace
        question = ' '.join(question.split())
        
        # Remove special characters but keep question marks
        question = re.sub(r'[^a-zA-Z0-9\s?]', '', question)
        
        # Ensure the question ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        return question

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into meaningful chunks based on common documentation patterns."""
        # Split on headers, numbered lists, etc.
        chunks = []
        current_chunk = []
        
        for line in text.split('\n'):
            # Start new chunk on headers or numbered steps
            if re.match(r'^[0-9]+\.|\b(Step|How to|Instructions):', line, re.I):
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _is_relevant_chunk(self, chunk: str) -> bool:
        """Determine if a chunk contains relevant information."""
        # Check minimum length
        if len(chunk.split()) < 10:
            return False
        
        # Check for instructional content
        instructional_patterns = [
            r'how to',
            r'steps? to',
            r'guide',
            r'tutorial',
            r'instructions?',
            r'[0-9]+\.',
            r'first',
            r'then',
            r'finally'
        ]
        
        return any(re.search(pattern, chunk.lower()) for pattern in instructional_patterns)

    def _process_chunk(self, chunk: str) -> str:
        """Process and format a chunk of text for better readability."""
        # Normalize whitespace
        chunk = ' '.join(chunk.split())
        
        # Format numbered steps
        chunk = re.sub(r'([0-9]+\.)', r'\n\1', chunk)
        
        # Format code blocks or examples
        chunk = re.sub(r'(`.*?`)', r'\n\1\n', chunk)
        
        return chunk.strip() 