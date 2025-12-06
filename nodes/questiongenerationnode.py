import os
import json
import re
import pandas as pd
import numpy as np
from tabulate import tabulate

# from openai_llm import OpenAILLM
from groq_llm import GroqLLM
from utils.state import STATE
# ===================== Data Understanding =====================

def data_understanding(df_profile: dict, model: str = "gemini-1.5-flash-latest") -> dict:  # FIXED MODEL NAME
    """
    Use HuggingFaceHub LLM to semantically understand the dataset domain and context.
    For text documents (PDFs, text files), analyzes actual content.
    For structured data, analyzes column names and characteristics.
    Returns structured understanding (domain, column_roles, use_cases, limitations).
    """
    data_type = df_profile.get('data_type', 'unknown')
    
    # Handle text documents differently (PDFs, text files)
    if data_type in ['pdf', 'text']:
        return _text_document_understanding(df_profile)
    else:
        return _structured_data_understanding(df_profile)


def _text_document_understanding(df_profile: dict) -> dict:
    """
    Analyze text documents (PDFs, text files) based on actual content.
    """
    try:
        # Initialize OpenAI LLM
        llm = GroqLLM(
            temperature=0.3,
            max_tokens=300
        )

        # Get sample content from the document
        sample_content = ""
        if 'sample' in df_profile and df_profile['sample']:
            # Extract text content from samples
            for sample in df_profile['sample'][:3]:  # Use first 3 samples
                if isinstance(sample, dict) and 'content' in sample:
                    sample_content += sample['content'][:500] + " "  # First 500 chars per sample
        
        # Truncate to manageable size for LLM
        sample_content = sample_content[:1500]  # Keep first 1500 characters
        
        prompt = f"""
        You are analyzing a text document. Based on the content below, determine the domain and purpose.

        Document info:
        - Type: {df_profile.get('data_type', 'text document')}
        - Pages/Chunks: {df_profile.get('n_rows', 'Unknown')}
        - Language: {df_profile.get('detected_languages', {})}
        
        Sample content:
        {sample_content}
        
        Based on this content, determine:
        1. Domain: What field/topic does this document cover? (technology, business, healthcare, education, research, etc.)
        2. Document type: What kind of document is this? (research paper, report, manual, etc.)
        3. Main topics: What are the key topics discussed?
        
        Be specific and base your analysis on the actual content.
        """

        if llm.is_available():  # Check if LLM is available
            response_text = llm(prompt)
            
            # Try to parse structured response, fallback to raw text
            domain = _extract_domain_from_content(sample_content, response_text)
            
            understanding = {
                "domain": domain,
                "analysis": response_text,
                "document_type": df_profile.get('data_type', 'text'),
                "content_length": len(sample_content),
                "chunks_count": df_profile.get('n_rows', 0)
            }
        else:
            return _fallback_text_understanding(df_profile, sample_content)

    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}. Using fallback analysis.")
        understanding = _fallback_text_understanding(df_profile, sample_content)

    # Pretty Console Output
    print(f"\n🤖 Data Understanding:")
    print(f"🌐 Domain: {understanding.get('domain', 'General')}")
    print(f"📊 Analysis: {understanding.get('analysis', 'Basic profile analysis')}")
    
    return understanding


def _structured_data_understanding(df_profile: dict) -> dict:
    """
    Analyze structured data (CSV, Excel, JSON) based on column names and characteristics.
    """
    try:
        # Initialize OpenAI LLM
        llm = GroqLLM(
            temperature=0.3,
            max_tokens=300
        )

        column_names = [col.get('name', 'Unknown') if isinstance(col, dict) else str(col) for col in df_profile.get('columns', [])[:10]]
        
        prompt = f"""
        You are a data expert analyzing any type of dataset. Do not assume a specific domain.

        Dataset profile summary:
        - Rows: {df_profile.get('n_rows', 'Unknown')}
        - Columns: {df_profile.get('n_cols', 'Unknown')}
        - Data type: {df_profile.get('data_type', 'Unknown')}
        - Column names: {column_names}

        Based ONLY on the column names and data characteristics, determine:
        1. Domain: What type of data is this? (can be ANY domain - sports, education, weather, social media, manufacturing, etc.)
        2. Main purpose: What might this data be used for?
        3. Key insights: What patterns could we find?

        Be data-driven and don't assume any default domain. If unclear, say "Mixed/General".
        """

        if llm.is_available():  # Check if LLM is available
            response_text = llm(prompt)
            
            # Try to parse structured response, fallback to raw text
            domain = "General"
            # Extract domain from response
            import re
            domain_match = re.search(r'Domain[:\s]*([^\n,]+)', response_text, re.IGNORECASE)
            if domain_match:
                domain = domain_match.group(1).strip()
            
            understanding = {
                "domain": domain,
                "analysis": response_text,
                "column_count": df_profile.get('n_cols', 0),
                "row_count": df_profile.get('n_rows', 0)
            }
        else:
            return _fallback_understanding(df_profile)

    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}. Using fallback analysis.")
        understanding = _fallback_understanding(df_profile)

    return understanding


def _extract_domain_from_content(content: str, llm_response: str) -> str:
    """
    Extract domain from document content and LLM response.
    Uses comprehensive keyword analysis to detect ANY domain.
    """
    content_lower = content.lower()
    
    # First try to extract from LLM response
    import re
    domain_match = re.search(r'Domain[:\s]*([^\n,]+)', llm_response, re.IGNORECASE)
    if domain_match:
        return domain_match.group(1).strip()
    
    # Expanded domain detection for any type of content
    domain_keywords = {
        "Technology/AI/ML": ['machine learning', 'automl', 'predictive analytics', 'algorithm', 'neural', 
                             'artificial intelligence', 'deep learning', 'data science', 'programming', 
                             'software', 'computer', 'technology', 'automation', 'digital'],
        "Academic/Research": ['research', 'study', 'analysis', 'methodology', 'findings', 'conclusion', 
                              'hypothesis', 'experiment', 'academic', 'scholar', 'publication', 
                              'literature review', 'theoretical', 'empirical'],
        "Business/Finance": ['business', 'market', 'strategy', 'revenue', 'customer', 'sales', 
                             'profit', 'finance', 'investment', 'economy', 'commercial', 'corporate', 
                             'management', 'leadership', 'marketing'],
        "Healthcare/Medical": ['health', 'medical', 'patient', 'treatment', 'clinical', 'diagnosis', 
                               'therapy', 'medicine', 'hospital', 'doctor', 'nurse', 'disease', 
                               'symptoms', 'healthcare', 'pharmaceutical'],
        "Education/Learning": ['education', 'learning', 'student', 'course', 'university', 'academic', 
                               'teaching', 'curriculum', 'school', 'training', 'pedagogy', 
                               'educational', 'instructor'],
        "Legal/Law": ['legal', 'law', 'court', 'judge', 'attorney', 'regulation', 'compliance', 
                      'legislation', 'contract', 'litigation', 'judicial', 'statute'],
        "Science/Engineering": ['science', 'engineering', 'physics', 'chemistry', 'biology', 'mathematics', 
                                'scientific', 'technical', 'laboratory', 'experiment', 'innovation'],
        "Arts/Culture": ['art', 'culture', 'creative', 'design', 'music', 'literature', 'history', 
                         'cultural', 'artistic', 'aesthetic', 'humanities'],
        "Sports/Recreation": ['sports', 'game', 'player', 'team', 'competition', 'athletic', 'fitness', 
                              'recreation', 'exercise', 'training', 'performance'],
        "Travel/Tourism": ['travel', 'tourism', 'destination', 'hotel', 'vacation', 'journey', 
                           'tourist', 'hospitality', 'adventure', 'exploration'],
        "Food/Nutrition": ['food', 'nutrition', 'recipe', 'cooking', 'diet', 'meal', 'restaurant', 
                           'culinary', 'ingredient', 'health food'],
        "Environment/Sustainability": ['environment', 'sustainability', 'climate', 'ecology', 'green', 
                                       'renewable', 'conservation', 'pollution', 'carbon', 'ecosystem'],
        "Government/Policy": ['government', 'policy', 'politics', 'public', 'administration', 'governance', 
                              'political', 'municipal', 'federal', 'regulation']
    }
    
    # Count matches for each domain
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Return the domain with highest score, or "General" if no matches
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        # Only return specific domain if we have reasonable confidence (at least 2 keyword matches)
        if domain_scores[best_domain] >= 2:
            return best_domain
    
    return "General"


def _fallback_text_understanding(df_profile: dict, content: str) -> dict:
    """
    Fallback text analysis when LLM is not available.
    """
    domain = _extract_domain_from_content(content, "")
    
    return {
        "domain": domain,
        "analysis": f"Text document with {df_profile.get('n_rows', 0)} chunks. Content analysis suggests {domain} domain based on keywords.",
        "document_type": df_profile.get('data_type', 'text'),
        "content_length": len(content),
        "chunks_count": df_profile.get('n_rows', 0),
        "fallback": True
    }


def _fallback_understanding(df_profile: dict) -> dict:
    """Fallback analysis when LLM is not available."""
    columns = df_profile.get('columns', [])
    data_type = df_profile.get('data_type', 'unknown')
    
    # Flexible heuristics for domain detection based on actual data
    column_names = []
    if isinstance(columns, list):
        for col in columns:
            if isinstance(col, dict):
                column_names.append(col.get('name', '').lower())
            else:
                column_names.append(str(col).lower())
    
    # Expanded domain detection - can handle ANY type of data
    domain = "Mixed/General"
    all_columns = ' '.join(column_names)
    
    # Sleep/Health domain (PRIORITY - check first)
    if any(word in all_columns for word in ['sleep', 'duration', 'quality', 'bedtime', 'wake', 'rest', 'dream']):
        domain = "Sleep/Health"
    # Healthcare domain (check early for health-related terms)
    elif any(word in all_columns for word in ['patient', 'diagnosis', 'treatment', 'medical', 'health', 'symptom', 'disease', 'stress level', 'bmi', 'blood pressure', 'heart rate']):
        domain = "Healthcare/Medical"
    # Financial/Business domain
    elif any(word in all_columns for word in ['price', 'cost', 'revenue', 'sales', 'profit', 'budget', 'income']):
        domain = "Business/Finance"
    # E-commerce/Retail domain
    elif any(word in all_columns for word in ['customer', 'product', 'order', 'purchase', 'item', 'cart', 'shipping']):
        domain = "Retail/E-commerce"
    # Education domain
    elif any(word in all_columns for word in ['student', 'grade', 'course', 'school', 'exam', 'score', 'teacher']):
        domain = "Education"
    # Sports domain
    elif any(word in all_columns for word in ['team', 'player', 'game', 'score', 'match', 'season', 'league', 'sport']):
        domain = "Sports"
    # Weather/Climate domain
    elif any(word in all_columns for word in ['temperature', 'weather', 'humidity', 'precipitation', 'wind', 'climate']):
        domain = "Weather/Climate"
    # Social Media domain
    elif any(word in all_columns for word in ['user', 'post', 'like', 'comment', 'share', 'follow', 'tweet', 'social']):
        domain = "Social Media"
    # Manufacturing/Industrial domain
    elif any(word in all_columns for word in ['machine', 'production', 'quality', 'defect', 'manufacturing', 'factory']):
        domain = "Manufacturing"
    # Transportation domain
    elif any(word in all_columns for word in ['vehicle', 'route', 'distance', 'speed', 'transport', 'travel', 'trip']):
        domain = "Transportation"
    
    return {
        "domain": domain,
        "analysis": f"Dataset contains {len(columns)} columns of type {data_type}. Basic analysis suggests {domain.lower()} domain.",
        "column_count": len(columns),
        "row_count": df_profile.get('n_rows', 0),
        "fallback": True
    }


# ===================== Question Generation Agent =====================
class QuestionGenerationAgent:
    def __init__(self, model: str = "gemini-1.5-flash-latest"):  # FIXED MODEL NAME
        
        self.model_name = model
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM with error handling."""
        try:
            print("🤖 Initializing OpenAI LLM for question generation...")
            self.llm = GroqLLM(
                temperature=0.4,  # Optimized for analytical creativity
                max_tokens=1500   # Higher limit for sophisticated responses
            )
            
            if self.llm.is_available():  # Check if LLM was successfully initialized
                print(f"✅ OpenAI LLM initialized successfully")
            else:
                print("⚠️ OpenAI LLM not available. Using fallback question generation.")
                self.llm = None
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI LLM: {e}. Using fallback question generation.")
            self.llm = None

    def _json_serialize(self, obj):
        """Helper to serialize numpy objects."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    def generate(self, profile: dict, understanding: dict, num_questions: int = 10):
        """
        Generate dataset-specific questions primarily via LLM.
        Only fallback to a small generic list if the LLM is unavailable or empty.
        """
        print(f"🧠 Starting question generation for {num_questions} questions...")
        print(f"📊 Profile data type: {profile.get('data_type', 'unknown')}")
        print(f"🎯 Domain: {understanding.get('domain', 'Unknown')}")

        # Check data type and route to appropriate generation method
        data_type = profile.get('data_type', 'unknown')
        
        # Route to text/PDF specific generation if applicable
        if data_type in ['pdf', 'text']:
            print(f"📄 Routing to text/PDF question generation for {data_type.upper()}...")
            return self._generate_text_questions(profile, understanding, num_questions)
        
        # If LLM is not available, immediately fallback
        if self.llm is None:
            print("⚠️ No LLM available, using fallback questions")
            questions = self._fallback_questions(profile, num_questions)
            questions = self._diversify_questions(questions, num_questions)
            print(f"✅ Fallback generated {len(questions)} questions")
            return questions

        # Build a concise, dataset-aware prompt for structured data
        
        columns = profile.get('columns', [])
        col_entries = []
        for col in columns[:20]:  # limit to first 20
            if isinstance(col, dict):
                name = str(col.get('name', 'unknown'))
                dtype = str(col.get('dtype', 'unknown'))
            else:
                name = str(col)
                dtype = 'unknown'
            col_entries.append(f"{name} ({dtype})" if dtype else name)

        dataset_summary = {
            'rows': profile.get('n_rows', 0),
            'columns': profile.get('n_cols', len(columns)),
            'data_type': profile.get('data_type', 'unknown'),
        }

        domain = understanding.get('domain', 'General')

        prompt = (
            "You are an expert data analyst. Based on the dataset profile below, "
            f"generate {num_questions} specific, data-driven questions that can be answered using this dataset. "
            "Each question must reference real column names and be analytically meaningful (comparisons, trends, distributions, relationships, rankings, etc.). "
            "Avoid generic questions that could apply to any dataset."
            "\n\nDataset Profile:\n"
            f"- Domain: {domain}\n"
            f"- Rows: {dataset_summary['rows']}\n"
            f"- Columns: {dataset_summary['columns']}\n"
            f"- Data Type: {dataset_summary['data_type']}\n"
            f"- Column list (name [dtype]): {', '.join(col_entries) if col_entries else 'N/A'}\n\n"
            "Output: Return ONLY a JSON array of strings (questions), no prose, no explanation."
        )

        try:
            print(f"🔄 Sending dataset-aware prompt to LLM (length: {len(prompt)} chars)")
            response_text = self.llm(prompt)
            if not response_text or len(response_text.strip()) == 0:
                raise ValueError("Empty LLM response")

            # Try JSON first
            questions = []
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, list):
                    questions = [q for q in parsed if isinstance(q, str) and len(q.strip()) > 0]
            except Exception:
                # Fallback: parse questions from text
                questions = self._parse_questions(response_text, num_questions)

            if not questions:
                print("⚠️ LLM returned no parseable questions; using fallback")
                questions = self._fallback_questions(profile, num_questions)

            questions = self._diversify_questions(questions, num_questions)
            print(f"🎯 LLM generation completed with {len(questions)} questions")
            return questions[:num_questions]

        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            questions = self._fallback_questions(profile, num_questions)
            questions = self._diversify_questions(questions, num_questions)
            print(f"✅ Fallback generated {len(questions)} questions")
            return questions[:num_questions]

    def _generate_text_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        """Generate questions specifically for text/PDF documents."""
        try:
            print(f"📝 Starting text/PDF question generation for {num_questions} questions...")
            
            # Get sample content
            sample_content = self._extract_text_content(profile)
            domain = understanding.get('domain', 'General')
            doc_type = profile.get('data_type', 'text document')
            chunks_count = profile.get('n_rows', 0)
            
            print(f"📄 Document info: {doc_type.upper()}, Domain: {domain}, Content: {len(sample_content)} chars, Chunks: {chunks_count}")
            
            # Create enhanced prompt for text analysis
            prompt = f"""You are an expert document analyst. Generate {num_questions} insightful analytical questions for this {doc_type} document.

Document Information:
- Domain: {domain}
- Document Type: {doc_type}
- Total Chunks/Pages: {chunks_count}
- Content Length: {len(sample_content)} characters

Sample Content:
{sample_content[:1200] if sample_content else 'No sample content available'}

Generate questions that would help analyze:
1. **Main Themes & Topics**: Key subjects and central ideas
2. **Arguments & Claims**: Core arguments, evidence, and reasoning
3. **Findings & Conclusions**: Key insights, results, and outcomes  
4. **Methodology & Approach**: How information is presented or research conducted
5. **Implications & Applications**: Practical uses and broader significance
6. **Context & Audience**: Intended readers and situational factors
7. **Quality & Credibility**: Source reliability and evidence strength
8. **Relationships & Patterns**: Connections between concepts and ideas

Requirements:
- Questions should be specific to the actual content shown
- Focus on analytical depth rather than basic comprehension
- Ensure questions can be answered from the document
- Make questions actionable for decision-making

Output only a JSON array of {num_questions} question strings, no other text."""
            
            # Try LLM generation first
            if self.llm and self.llm.is_available():
                print(f"🤖 Sending prompt to OpenAI LLM (length: {len(prompt)} chars)")
                response = self.llm(prompt)
                
                if response and len(response.strip()) > 0:
                    print(f"✅ Received LLM response (length: {len(response)} chars)")
                    
                    # Try JSON parsing first
                    questions = []
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, list):
                            questions = [q for q in parsed if isinstance(q, str) and len(q.strip()) > 10]
                    except json.JSONDecodeError:
                        print("⚠️ JSON parsing failed, trying text parsing...")
                        questions = self._parse_questions(response, num_questions)
                    
                    if questions:
                        print(f"🎉 Successfully parsed {len(questions)} questions from LLM")
                        
                        # Supplement with fallbacks if needed
                        if len(questions) < num_questions:
                            needed = num_questions - len(questions)
                            print(f"🔄 Supplementing with {needed} fallback questions...")
                            fallbacks = self._fallback_text_questions(profile, sample_content, needed)
                            questions.extend(fallbacks)
                        
                        final_questions = questions[:num_questions]
                        print(f"✅ Text question generation completed: {len(final_questions)} questions")
                        return final_questions
                    else:
                        print("⚠️ No valid questions parsed from LLM response")
                else:
                    print("⚠️ Empty or invalid LLM response")
            
            # Fallback to text-specific questions
            print("🔄 Using fallback text question generation...")
            questions = self._fallback_text_questions(profile, sample_content, num_questions)
            print(f"✅ Fallback generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            print(f"❌ Text question generation failed: {e}")
            print(f"🔄 Using emergency fallback questions...")
            return self._fallback_text_questions(profile, "", num_questions)
       
    def _extract_text_content(self, profile: dict) -> str:
        """Extract text content from document profile with improved handling."""
        sample_content = ""
        
        try:
            # Check multiple possible locations for content
            if 'sample' in profile and profile['sample']:
                samples_used = 0
                for sample in profile['sample'][:5]:  # Use up to 5 samples for better coverage
                    if isinstance(sample, dict):
                        # Try different content fields
                        content = sample.get('content') or sample.get('text') or sample.get('page_content')
                        if content and isinstance(content, str) and len(content.strip()) > 20:
                            # Clean and add content
                            clean_content = content.strip()[:600]
                            sample_content += clean_content + "\n\n"
                            samples_used += 1
                    elif isinstance(sample, str) and len(sample.strip()) > 20:
                        # Direct string content
                        sample_content += sample.strip()[:600] + "\n\n"
                        samples_used += 1
                
                print(f"📄 Extracted content from {samples_used} document chunks")
            
            # Also check for direct content fields in profile
            elif 'content' in profile and isinstance(profile['content'], str):
                sample_content = profile['content'][:1800]
                print("📄 Extracted content from profile content field")
            
            elif 'text' in profile and isinstance(profile['text'], str):
                sample_content = profile['text'][:1800]
                print("📄 Extracted content from profile text field")
            
            # Final cleanup and limit
            if sample_content:
                sample_content = sample_content.strip()[:2000]  # Increased limit for better context
                print(f"📊 Final content length: {len(sample_content)} characters")
            else:
                print("⚠️ No content could be extracted from document profile")
                sample_content = "No content available for analysis"
            
            return sample_content
            
        except Exception as e:
            print(f"⚠️ Error extracting text content: {e}")
            return "Error extracting document content"


    def _generate_structured_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        """
        Generate sophisticated, domain-specific analytical questions for structured data using OpenAI-OS-120B.
        """
        try:
            domain = understanding.get('domain', 'General')
            
            # Analyze the dataset structure deeply
            dataset_analysis = self._analyze_dataset_structure(profile)
            
            # Create expert-level prompt for sophisticated question generation
            prompt = self._create_expert_analysis_prompt(domain, dataset_analysis, num_questions)
            
            print(f"🧠 Sending expert-level prompt to OpenAI-OS-120B (length: {len(prompt)} chars)")
            response_text = self.llm(prompt)
            
            if not response_text or len(response_text.strip()) == 0:
                print("⚠️ LLM returned empty response! Using enhanced fallback.")
                return self._create_expert_fallback_questions(domain, dataset_analysis, num_questions)
            
            print(f"✅ Expert-level response received (length: {len(response_text)} chars)")
            
            # Extract and validate sophisticated questions
            questions = self._extract_analytical_questions(response_text, num_questions)
            
            # If we don't have enough quality questions, supplement with expert fallbacks
            if len(questions) < num_questions:
                expert_fallbacks = self._create_expert_fallback_questions(
                    domain, dataset_analysis, num_questions - len(questions)
                )
                questions.extend(expert_fallbacks)
            
            return questions[:num_questions]

        except Exception as e:
            print(f"⚠️ Expert question generation failed: {e}. Using expert fallback.")
            return self._create_expert_fallback_questions(domain, self._analyze_dataset_structure(profile), num_questions)

    def _analyze_dataset_structure(self, profile: dict) -> dict:
        """
        Deeply analyze dataset structure to understand the type of data and analytical opportunities.
        """
        columns = profile.get('columns', [])
        data_type = profile.get('data_type', 'unknown')
        n_rows = profile.get('n_rows', 0)
        n_cols = profile.get('n_cols', 0)
        
        analysis = {
            'data_type': data_type,
            'size': {'rows': n_rows, 'columns': n_cols},
            'column_types': {'numeric': [], 'categorical': [], 'temporal': [], 'text': []},
            'key_indicators': [],
            'time_series_potential': False,
            'analytical_dimensions': []
        }
        
        # Analyze each column in detail
        for col in columns:
            if isinstance(col, dict):
                col_name = col.get('name', '').lower()
                col_type = col.get('dtype', '').lower()
                
                # Categorize columns
                if 'int' in col_type or 'float' in col_type:
                    analysis['column_types']['numeric'].append(col.get('name', ''))
                elif 'object' in col_type:
                    # Detect if it's actually temporal data
                    if any(word in col_name for word in ['date', 'time', 'year', 'month', 'quarter', 'period']):
                        analysis['column_types']['temporal'].append(col.get('name', ''))
                        analysis['time_series_potential'] = True
                    else:
                        analysis['column_types']['categorical'].append(col.get('name', ''))
                elif 'datetime' in col_type:
                    analysis['column_types']['temporal'].append(col.get('name', ''))
                    analysis['time_series_potential'] = True
                
                # Identify key economic/financial indicators
                if any(word in col_name for word in ['gdp', 'growth', 'rate', 'index', 'price', 'revenue', 'sales', 'profit']):
                    analysis['key_indicators'].append(col.get('name', ''))
        
        # Determine analytical dimensions based on data structure
        if analysis['time_series_potential']:
            analysis['analytical_dimensions'].extend(['temporal_trends', 'seasonality', 'cyclical_patterns', 'volatility_analysis'])
        
        if len(analysis['column_types']['numeric']) >= 2:
            analysis['analytical_dimensions'].extend(['correlation_analysis', 'comparative_analysis'])
        
        if len(analysis['column_types']['categorical']) >= 1:
            analysis['analytical_dimensions'].extend(['segmentation_analysis', 'distribution_analysis'])
        
        return analysis
    
    def _create_expert_analysis_prompt(self, domain: str, dataset_analysis: dict, num_questions: int) -> str:
        """
        Create expert-level prompts for sophisticated analytical question generation.
        """
        # Build comprehensive context
        context_parts = [
            f"**Dataset Profile:**",
            f"- Domain: {domain}",
            f"- Size: {dataset_analysis['size']['rows']:,} records × {dataset_analysis['size']['columns']} variables",
            f"- Data Type: {dataset_analysis['data_type']}"
        ]
        
        # Add column analysis
        if dataset_analysis['column_types']['numeric']:
            context_parts.append(f"- Numeric Variables ({len(dataset_analysis['column_types']['numeric'])}): {', '.join(dataset_analysis['column_types']['numeric'][:5])}")  
        
        if dataset_analysis['column_types']['temporal']:
            context_parts.append(f"- Temporal Variables: {', '.join(dataset_analysis['column_types']['temporal'])}")
            
        if dataset_analysis['column_types']['categorical']:    
            context_parts.append(f"- Categorical Variables ({len(dataset_analysis['column_types']['categorical'])}): {', '.join(dataset_analysis['column_types']['categorical'][:5])}")
        
        if dataset_analysis['key_indicators']:
            context_parts.append(f"- Key Performance Indicators: {', '.join(dataset_analysis['key_indicators'][:3])}")
        
        context_parts.append(f"- Analytical Capabilities: {', '.join(dataset_analysis['analytical_dimensions'])}")
        
        # Create domain-specific expert prompt
        if 'economic' in domain.lower() or 'gdp' in str(dataset_analysis['key_indicators']).lower():
            return self._create_economics_expert_prompt(context_parts, num_questions)
        elif 'health' in domain.lower() or 'medical' in domain.lower():
            return self._create_healthcare_expert_prompt(context_parts, num_questions)
        elif 'business' in domain.lower() or 'finance' in domain.lower():
            return self._create_business_expert_prompt(context_parts, num_questions)
        elif 'social' in domain.lower() or 'media' in domain.lower():
            return self._create_social_media_expert_prompt(context_parts, num_questions)
        else:
            return self._create_general_expert_prompt(context_parts, num_questions)
    
    def _create_economics_expert_prompt(self, context_parts: list, num_questions: int) -> str:
        """
        Create expert-level economics/macroeconomics analytical questions.
        """
        context = "\n".join(context_parts)
        
        return f"""You are a senior macroeconomist and data analyst with expertise in economic policy analysis, growth modeling, and business cycle research.

Analyze this economic dataset and generate {num_questions} sophisticated analytical questions that would be asked by economic researchers, policy makers, and financial analysts.

{context}

**Requirements for Questions:**
1. **Policy-Relevant**: Questions that inform economic policy decisions
2. **Technically Sophisticated**: Use proper economic terminology and concepts
3. **Actionable**: Lead to insights that can guide investment/policy decisions
4. **Multi-dimensional**: Consider temporal, sectoral, and cyclical dimensions
5. **Empirically Testable**: Can be answered through rigorous data analysis

**Focus Areas:**
- **Growth Dynamics**: Long-term trends, cycles, structural breaks
- **Business Cycle Analysis**: Recession identification, recovery patterns
- **Sectoral Analysis**: Component contributions, structural shifts
- **External Sector**: Trade balance dynamics, external shocks impact
- **Volatility & Risk**: Measure economic stability and vulnerability
- **Comparative Analysis**: Cross-period, cross-component comparisons
- **Leading Indicators**: Predictive patterns and early warning signals
- **Policy Evaluation**: Impact assessment of economic interventions

**Question Format**: Each question should be specific, measurable, and directly answerable from the dataset.

**Example Quality Level**:
- "During which periods did New Zealand experience two or more consecutive quarters of negative real GDP growth (q/q%)?" 
- "How has the trade balance (exports − imports) fluctuated, and how does it correlate with overall GDP performance?"
- "Which expenditure component has contributed most to GDP volatility during economic downturns?"

Generate {num_questions} questions of this caliber:"""
    
    def _extract_analytical_questions(self, response_text: str, num_questions: int) -> list:
        """
        Extract sophisticated analytical questions from OpenAI-OS-120B response with improved parsing.
        """
        questions = []
        
        print(f"🔍 Parsing expert-level response for sophisticated questions...")
        
        # Enhanced patterns for sophisticated question extraction
        patterns = [
            r'(?:^|\n)\s*\d+\.\s*([^\n]{20,}\?)\s*(?=\n|$)',  # Numbered questions
            r'(?:^|\n)\s*-\s*([^\n]{20,}\?)\s*(?=\n|$)',      # Bullet point questions
            r'(?:^|\n)\s*\*\s*([^\n]{20,}\?)\s*(?=\n|$)',     # Asterisk bullet questions
            r'(?:^|\n)\s*([A-Z][^\n]{20,}\?)\s*(?=\n|$)',     # Questions starting with capital
            r'"([^"]{20,}\?)"',                               # Quoted questions
            r'(?:Question|Q)\s*\d*:?\s*([^\n]{20,}\?)',       # Explicit question markers
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            print(f"🔍 Pattern {i+1} found {len(matches)} matches")
            
            for match in matches:
                clean_question = self._clean_question(match)
                
                # Quality filters for sophisticated questions
                if (len(clean_question) > 20 and 
                    clean_question not in questions and 
                    self._is_sophisticated_question(clean_question)):
                    questions.append(clean_question)
                    print(f"✅ Added sophisticated question: {clean_question[:80]}...")
                    if len(questions) >= num_questions:
                        break
            if len(questions) >= num_questions:
                break
        
        print(f"📊 Final sophisticated questions count: {len(questions)}")
        return questions
    
    def _clean_question(self, question: str) -> str:
        """
        Clean and standardize question format.
        """
        # Remove leading numbers, bullets, quotes
        question = re.sub(r'^\d+\.\s*|^[-\*]\s*|^"|"$', '', question.strip())
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        return question.strip()
    
    def _is_sophisticated_question(self, question: str) -> bool:
        """
        Check if question meets sophistication criteria.
        """
        question_lower = question.lower()
        
        # Sophisticated question indicators
        sophisticated_terms = [
            'correlation', 'trend', 'pattern', 'cycle', 'volatility', 'impact',
            'during which', 'how has', 'what patterns', 'which periods',
            'contributed most', 'fluctuate', 'compare', 'versus', 'relationship',
            'analysis', 'dynamics', 'evolution', 'components', 'factors'
        ]
        
        # Reject overly generic questions
        generic_phrases = [
            'what is', 'tell me about', 'describe', 'explain the basic',
            'what are some', 'give me', 'list the', 'show me'
        ]
        
        # Check for sophistication
        has_sophisticated_terms = any(term in question_lower for term in sophisticated_terms)
        is_not_generic = not any(phrase in question_lower for phrase in generic_phrases)
        
        return has_sophisticated_terms and is_not_generic and len(question) > 30
    
    def _create_expert_fallback_questions(self, domain: str, dataset_analysis: dict, num_questions: int) -> list:
        """
        Generate expert-level fallback questions when LLM fails, tailored to specific domains.
        """
        if 'economic' in domain.lower() or any('gdp' in str(indicator).lower() for indicator in dataset_analysis.get('key_indicators', [])):
            return self._create_economics_fallback_questions(dataset_analysis, num_questions)
        elif 'health' in domain.lower() or 'medical' in domain.lower():
            return self._create_healthcare_fallback_questions(dataset_analysis, num_questions)
        elif 'business' in domain.lower() or 'finance' in domain.lower():
            return self._create_business_fallback_questions(dataset_analysis, num_questions)
        else:
            return self._create_general_analytical_fallback_questions(dataset_analysis, num_questions)
    
    def _create_economics_fallback_questions(self, dataset_analysis: dict, num_questions: int) -> list:
        """
        Create diverse, sophisticated economics questions with varied visualization needs.
        """
        # Mix of different question types to ensure visualization diversity
        diverse_economics_questions = [
            # Statistical/Calculation questions (bar charts, line charts)
            "How has real GDP growth (y/y%) evolved over time, and what are the identifiable long-term growth cycles?",
            "Which expenditure component has contributed most to GDP growth volatility over the analysis period?",
            "What is the average quarterly growth rate for each major economic component?",
            
            # Comparative questions (bar charts, grouped charts)
            "During which periods did the economy experience two or more consecutive quarters of negative real GDP growth?",
            "How does private consumption compare to government consumption in terms of volatility and trend?",
            "Which quarters show the highest and lowest economic performance across all indicators?",
            
            # Correlation/Relationship questions (scatter plots, heatmaps)
            "How has the trade balance (exports − imports) fluctuated, and what is its correlation with GDP performance?",
            "What relationships exist between investment levels and subsequent GDP growth patterns?",
            "How do external sector dynamics correlate with domestic economic indicators?",
            
            # Distribution questions (histograms, box plots)
            "What is the distribution of quarterly growth rates across different economic components?",
            "Which economic indicators show the most consistent performance versus high volatility?",
            "What seasonal patterns exist in quarterly GDP data across different sectors?",
            
            # Trend analysis questions (line charts, time series)
            "How have exports and imports tracked relative to each other over the analysis period?",
            "What long-term structural changes are evident in the composition of GDP?",
            "How has the share of government spending evolved relative to private sector activity?",
            
            # Complex analytical questions (multiple chart types)
            "What are the leading indicators of economic downturns based on component relationships?",
            "How do nominal versus real measures reveal inflationary pressures across periods?",
            "What economic sectors show the strongest predictive power for overall GDP performance?",
            
            # Insight/Pattern questions (varied visualizations)
            "What cyclical patterns emerge when analyzing GDP components over multiple quarters?",
            "Which economic indicators provide the earliest warning signals of economic shifts?"
        ]
        
        return diverse_economics_questions[:num_questions]
    
    def _create_general_expert_prompt(self, context_parts: list, num_questions: int) -> str:
        """
        Create expert-level general analytical questions for any domain.
        """
        context = "\n".join(context_parts)
        
        return f"""You are a senior data analyst with expertise in advanced analytics across multiple domains.

Analyze this dataset and generate {num_questions} sophisticated analytical questions that would provide deep business insights.

{context}

**Requirements for Questions:**
1. **Analytically Sophisticated**: Use advanced statistical and analytical concepts
2. **Business-Relevant**: Questions that drive strategic decision-making
3. **Actionable**: Lead to insights that can guide policy/business decisions
4. **Multi-dimensional**: Consider temporal, categorical, and quantitative dimensions
5. **Data-Driven**: Can be answered through rigorous data analysis

**Focus Areas:**
- **Trend Analysis**: Long-term patterns, cycles, structural changes
- **Segmentation Analysis**: Group comparisons, demographic patterns
- **Predictive Insights**: Leading indicators, forecasting opportunities
- **Risk Assessment**: Volatility patterns, anomaly detection
- **Correlation Analysis**: Variable relationships, causality assessment
- **Performance Optimization**: Efficiency metrics, improvement opportunities
- **Comparative Analysis**: Cross-period, cross-segment comparisons
- **Strategic Planning**: Data-driven recommendations, scenario analysis

**Question Format**: Each question should be specific, measurable, and directly answerable from the dataset.

**Example Quality Level**:
- "What are the most significant temporal trends and how do they correlate with external factors?"
- "Which variables demonstrate the strongest predictive power for key performance outcomes?"
- "How do different categorical segments compare in terms of variability and central tendencies?"

Generate {num_questions} questions of this caliber:"""
    
    def _create_healthcare_fallback_questions(self, dataset_analysis: dict, num_questions: int) -> list:
        """
        Create sophisticated healthcare analysis questions.
        """
        healthcare_questions = [
            "Which patient demographics and clinical characteristics are most predictive of treatment outcomes?",
            "How do comorbidity patterns affect treatment efficacy across different patient populations?",
            "What are the optimal intervention thresholds based on risk stratification analysis?",
            "Which biomarkers show the strongest correlation with disease progression over time?",
            "How do treatment adherence patterns vary across demographic segments and what factors drive compliance?",
            "What seasonal or temporal patterns exist in disease incidence and how do they correlate with environmental factors?",
            "Which combination of risk factors creates the highest probability of adverse outcomes?",
            "How do healthcare resource utilization patterns differ across patient severity levels?",
            "What are the predictive indicators for patient readmission within 30, 60, and 90 days?",
            "How do treatment costs correlate with patient outcomes and what drives cost-effectiveness optimization?"
        ]
        
        return healthcare_questions[:num_questions]
    
    def _create_business_fallback_questions(self, dataset_analysis: dict, num_questions: int) -> list:
        """
        Create sophisticated business analysis questions.
        """
        business_questions = [
            "Which customer segments demonstrate the highest lifetime value and what characteristics define them?",
            "How do market dynamics and competitive pressures impact pricing elasticity across product categories?",
            "What are the leading indicators of customer churn and which intervention strategies are most effective?",
            "How do seasonal demand patterns vary across geographic regions and product lines?",
            "Which operational metrics show the strongest correlation with financial performance outcomes?",
            "What is the optimal resource allocation strategy based on ROI analysis across business units?",
            "How do external market conditions influence internal operational efficiency metrics?",
            "Which customer acquisition channels provide the best cost-per-acquisition and retention rates?",
            "What predictive patterns exist in sales data that can inform inventory optimization strategies?",
            "How do employee performance metrics correlate with customer satisfaction and revenue generation?"
        ]
        
        return business_questions[:num_questions]
    
    def _create_general_analytical_fallback_questions(self, dataset_analysis: dict, num_questions: int) -> list:
        """
        Create sophisticated general analytical questions for any domain.
        """
        general_questions = [
            "What are the most significant temporal trends and how do they correlate with external factors?",
            "Which variables demonstrate the strongest predictive power for key performance outcomes?",
            "How do different categorical segments compare in terms of variability and central tendencies?",
            "What cyclical or seasonal patterns exist and what factors drive these recurring behaviors?",
            "Which combinations of variables create the highest risk or opportunity scenarios?",
            "What are the leading indicators that precede significant changes in primary metrics?",
            "How do outliers and anomalies provide insights into underlying system behaviors?",
            "What correlation structures exist between variables and how stable are these relationships over time?",
            "Which factors contribute most to overall system volatility and uncertainty?",
            "How can predictive modeling be optimized based on the identified patterns and relationships?"
        ]
        
        return general_questions[:num_questions]
    
    def _parse_questions(self, response_text: str, num_questions: int) -> list:
        """Extract questions from LLM response with improved parsing."""
        questions = []
        
        print(f"🔍 Parsing LLM response for questions...")
        print(f"📝 Raw response length: {len(response_text)} chars")
        print(f"📝 First 200 chars: {response_text[:200]}...")
        
        # Enhanced patterns for question extraction
        patterns = [
            r'\d+\.\s*(.+?\?)',  # 1. Question?
            r'(?:^|\n)\s*([^?\n]{10,}\?)',  # Any line ending with ? (at least 10 chars)
            r'(?:Question|Q)\s*\d*:?\s*(.+?\?)',  # Question: text?
            r'(?:^|\n)\s*[-•*]\s*(.+?\?)',  # Bullet points with ?
            r'(?:^|\n)([A-Z][^?\n]{10,}\?)',  # Lines starting with capital, ending with ?
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            print(f"🔍 Pattern {i+1} found {len(matches)} matches")
            
            for match in matches:
                clean_question = match.strip()
                # Remove leading numbers, bullets, etc.
                clean_question = re.sub(r'^\d+\.\s*|^[-•*]\s*', '', clean_question)
                
                if len(clean_question) > 10 and clean_question not in questions:
                    questions.append(clean_question)
                    print(f"✅ Added question: {clean_question[:60]}...")
                    if len(questions) >= num_questions:
                        break
            if len(questions) >= num_questions:
                break
        
        # If no questions found with patterns, try simple line-by-line extraction
        if not questions:
            print("⚠️ No pattern matches. Trying line-by-line extraction...")
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and ('?' in line or any(word in line.lower() for word in ['what', 'how', 'why', 'which', 'where', 'when'])):
                    # Clean up the line
                    clean_line = re.sub(r'^\d+\.\s*|^[-•*]\s*', '', line)
                    if clean_line not in questions:
                        questions.append(clean_line if clean_line.endswith('?') else clean_line + '?')
                        print(f"✅ Added from line: {clean_line[:60]}...")
                        if len(questions) >= num_questions:
                            break
        
        print(f"📊 Final parsed questions count: {len(questions)}")
        return questions

    def _question_type(self, q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ['compare', 'versus', 'vs', 'rank', 'top', 'bottom', 'highest', 'lowest']):
            return 'comparative'
        if any(k in ql for k in ['correlation', 'relationship', 'associate', 'relate']):
            return 'correlation'
        if any(k in ql for k in ['distribution', 'spread', 'range', 'histogram', 'variance', 'std']):
            return 'distribution'
        if any(k in ql for k in ['trend', 'over time', 'season', 'change over', 'evolve']):
            return 'trend'
        if any(k in ql for k in ['predict', 'forecast', 'future']):
            return 'predictive'
        if any(k in ql for k in ['mean', 'average', 'median', 'sum', 'total', 'count']):
            return 'statistical'
        return 'analytical'

    def _diversify_questions(self, questions: list, limit: int) -> list:
        """Ensure a diverse set of question types and remove near-duplicates"""
        # Deduplicate (case-insensitive)
        seen = set()
        unique = []
        for q in questions:
            key = re.sub(r'\s+', ' ', q.strip().lower())
            if key not in seen:
                seen.add(key)
                unique.append(q.strip())
        
        # Bucket by type
        buckets = {
            'statistical': [], 'comparative': [], 'correlation': [],
            'distribution': [], 'trend': [], 'analytical': [], 'predictive': []
        }
        for q in unique:
            buckets[self._question_type(q)].append(q)
        
        # Round-robin selection across buckets to maximize variety
        order = ['comparative','correlation','distribution','trend','statistical','analytical','predictive']
        diversified = []
        idx = 0
        while len(diversified) < min(limit, len(unique)):
            added = False
            for t in order:
                if idx < len(buckets[t]):
                    diversified.append(buckets[t][idx])
                    if len(diversified) >= limit:
                        break
                    added = True
            if not added:
                break
            idx += 1
        
        # If still short, append remaining unique questions
        if len(diversified) < limit:
            for q in unique:
                if q not in diversified:
                    diversified.append(q)
                if len(diversified) >= limit:
                    break
        
        return diversified[:limit]
    
    def _create_domain_specific_prompt(self, domain: str, profile: dict, column_details: list, 
                                      numeric_cols: list, categorical_cols: list, 
                                      date_cols: list, num_questions: int) -> str:
        """Create sophisticated domain-specific prompts for different data types"""
        
        base_info = f"""
Dataset Information:
- Domain: {domain}
- Records: {profile.get('n_rows', 0):,}
- Features: {profile.get('n_cols', 0)}
- Data Types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(date_cols)} temporal

Key Columns:
{chr(10).join(column_details[:8])}
"""
        
        # Domain-specific question frameworks
        if 'social media' in domain.lower() or 'sentiment' in domain.lower():
            prompt = f"""{base_info}

As a Social Media & Sentiment Analysis expert, generate {num_questions} sophisticated analytical questions that would provide actionable business insights:

Focus Areas:
1. **Sentiment Patterns**: How sentiment varies across different dimensions
2. **Behavioral Analysis**: User engagement and interaction patterns  
3. **Content Performance**: What drives positive vs negative sentiment
4. **Temporal Trends**: How sentiment changes over time periods
5. **Predictive Modeling**: Forecasting sentiment and engagement
6. **Business Impact**: Actionable insights for brand management
7. **Demographic Insights**: How different user groups respond
8. **Content Strategy**: Optimization opportunities

Generate questions that go beyond basic statistics to provide strategic insights:"""

        elif 'airline' in domain.lower() or 'transport' in domain.lower():
            prompt = f"""{base_info}

As an Aviation Industry Analyst, generate {num_questions} strategic analytical questions for airline performance optimization:

Analytical Frameworks:
1. **Customer Experience**: Service quality and satisfaction drivers
2. **Operational Efficiency**: Route, schedule, and resource optimization
3. **Revenue Management**: Pricing strategies and demand patterns
4. **Risk Assessment**: Safety, weather, and operational risk factors
5. **Market Intelligence**: Competitive positioning and market share
6. **Predictive Analytics**: Demand forecasting and capacity planning
7. **Cost Optimization**: Fuel, maintenance, and operational costs
8. **Regulatory Compliance**: Safety and regulatory performance

Generate strategic questions that airline executives would ask:"""

        elif 'e-commerce' in domain.lower() or 'retail' in domain.lower():
            prompt = f"""{base_info}

As an E-commerce Analytics Expert, generate {num_questions} business-critical questions for retail optimization:

Business Intelligence Areas:
1. **Customer Segmentation**: Behavior-based customer grouping
2. **Product Performance**: Best/worst performing products and categories
3. **Sales Optimization**: Pricing, promotion, and inventory strategies
4. **Geographic Analysis**: Regional performance and expansion opportunities
5. **Seasonal Patterns**: Time-based trends and forecasting
6. **Customer Lifetime Value**: Retention and loyalty insights
7. **Market Basket Analysis**: Cross-selling and bundling opportunities
8. **Operational Efficiency**: Supply chain and fulfillment optimization

Generate questions that drive revenue growth and operational excellence:"""

        elif 'health' in domain.lower() or 'medical' in domain.lower():
            prompt = f"""{base_info}

As a Healthcare Data Scientist, generate {num_questions} clinically-relevant analytical questions:

Clinical Research Areas:
1. **Risk Stratification**: Identifying high-risk patient populations
2. **Treatment Efficacy**: Measuring intervention effectiveness
3. **Predictive Modeling**: Disease progression and outcome prediction
4. **Population Health**: Demographic and geographic health patterns
5. **Resource Optimization**: Healthcare utilization and capacity planning
6. **Quality Metrics**: Care quality indicators and benchmarking
7. **Cost Analysis**: Healthcare economics and value-based care
8. **Preventive Care**: Early intervention and screening strategies

Generate questions that improve patient outcomes and healthcare delivery:"""

        elif 'finance' in domain.lower() or 'business' in domain.lower():
            prompt = f"""{base_info}

As a Financial Analyst, generate {num_questions} strategic financial questions for business intelligence:

Financial Analysis Dimensions:
1. **Profitability Analysis**: Margin optimization and cost management
2. **Risk Assessment**: Financial risk factors and mitigation strategies
3. **Performance Metrics**: KPI tracking and benchmark analysis
4. **Market Analysis**: Competitive positioning and market dynamics
5. **Forecasting**: Revenue, expense, and cash flow predictions
6. **Investment Analysis**: ROI and capital allocation decisions
7. **Customer Economics**: Customer acquisition cost and lifetime value
8. **Operational Efficiency**: Process optimization and automation opportunities

Generate questions that drive financial performance and strategic decision-making:"""

        else:
            # Generic sophisticated prompt for any domain
            prompt = f"""{base_info}

As a Senior Data Analyst, generate {num_questions} sophisticated analytical questions that would provide deep business insights:

Analytical Approaches:
1. **Descriptive Analytics**: What patterns exist in the current data?
2. **Diagnostic Analytics**: Why are these patterns occurring?
3. **Predictive Analytics**: What is likely to happen in the future?
4. **Prescriptive Analytics**: What actions should be taken?
5. **Segmentation Analysis**: How do different groups compare?
6. **Trend Analysis**: How are metrics changing over time?
7. **Correlation Analysis**: What relationships exist between variables?
8. **Anomaly Detection**: What outliers or unusual patterns exist?

Generate questions that go beyond basic statistics to provide actionable insights:"""
        
        return prompt
    
    def _enhanced_fallback_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        """Generate sophisticated domain-specific fallback questions"""
        domain = understanding.get('domain', 'General')
        columns = profile.get('columns', [])
        
        # Extract column information
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for col in columns:
            if isinstance(col, dict):
                col_name = col.get('name', 'Unknown')
                col_type = col.get('dtype', 'Unknown')
                
                if 'int' in col_type.lower() or 'float' in col_type.lower():
                    numeric_cols.append(col_name)
                elif 'object' in col_type.lower():
                    categorical_cols.append(col_name)
                elif 'datetime' in col_type.lower():
                    date_cols.append(col_name)
        
        questions = []
        
        # Domain-specific sophisticated questions
        if 'social media' in domain.lower() or 'sentiment' in domain.lower():
            domain_questions = [
                "Which factors are most predictive of negative sentiment, and how can they be mitigated?",
                "How does sentiment distribution vary across different user segments or demographics?",
                "What are the key drivers of high-confidence sentiment predictions vs low-confidence ones?",
                "Can we identify sentiment patterns that indicate emerging issues or viral trends?",
                "What is the relationship between sentiment confidence and actual user engagement?",
                "How do different types of negative feedback (complaints, service issues, etc.) cluster together?",
                "What temporal patterns exist in sentiment that could inform proactive response strategies?",
                "Which combination of factors creates the highest risk for reputation damage?"
            ]
            
        elif 'e-commerce' in domain.lower() or 'retail' in domain.lower():
            domain_questions = [
                "What customer segments show the highest lifetime value and retention rates?",
                "Which product categories have the most price sensitivity and seasonal variation?",
                "How do geographic regions differ in purchasing behavior and preferences?",
                "What are the optimal inventory levels to minimize stockouts while reducing carrying costs?",
                "Which marketing channels and campaigns generate the highest ROI?",
                "What customer behavior patterns indicate churn risk or upselling opportunities?",
                "How do order patterns and timing affect fulfillment costs and customer satisfaction?",
                "What product bundling strategies would maximize average order value?"
            ]
            
        elif 'health' in domain.lower() or 'medical' in domain.lower():
            domain_questions = [
                "Which patient characteristics are most predictive of treatment success or complications?",
                "How do different demographic groups respond to various treatment protocols?",
                "What early warning indicators can predict disease progression or deterioration?",
                "Which interventions have the highest impact on patient outcomes relative to cost?",
                "How do comorbidities and risk factors interact to affect treatment effectiveness?",
                "What patterns exist in healthcare utilization that could improve resource allocation?",
                "Which quality metrics are most correlated with patient satisfaction and outcomes?",
                "How can predictive models improve preventive care and early intervention strategies?"
            ]
            
        elif 'finance' in domain.lower() or 'business' in domain.lower():
            domain_questions = [
                "What are the key drivers of profitability variation across different business segments?",
                "Which financial metrics are most predictive of future performance and growth?",
                "How do market conditions and external factors impact financial performance?",
                "What customer segments provide the highest margins and lowest acquisition costs?",
                "Which operational inefficiencies represent the largest cost-saving opportunities?",
                "How do pricing strategies affect demand elasticity across different market segments?",
                "What leading indicators can predict cash flow challenges or opportunities?",
                "Which investment priorities would generate the highest returns and strategic value?"
            ]
            
        else:
            # Generic sophisticated questions
            domain_questions = [
                "What are the most significant patterns and anomalies that require immediate attention?",
                "Which variables have the strongest predictive power for key outcomes?",
                "How do different segments or groups compare across critical performance metrics?",
                "What temporal trends suggest opportunities for optimization or intervention?",
                "Which combinations of factors create the highest risk or opportunity scenarios?",
                "What correlations and relationships provide actionable insights for decision-making?",
                "How can predictive models be used to improve planning and resource allocation?",
                "What data quality issues or gaps could be affecting analysis reliability?"
            ]
        
        # Add domain-specific questions
        questions.extend(domain_questions[:num_questions])
        
        # Add column-specific analytical questions if needed
        if len(questions) < num_questions and numeric_cols:
            for col in numeric_cols[:3]:
                if len(questions) >= num_questions:
                    break
                questions.append(f"What factors drive the variability in {col}, and how can it be optimized?")
                questions.append(f"Are there any unexpected patterns or outliers in {col} that warrant investigation?")
        
        if len(questions) < num_questions and categorical_cols:
            for col in categorical_cols[:2]:
                if len(questions) >= num_questions:
                    break
                questions.append(f"How does {col} segmentation impact key performance metrics and outcomes?")
        
        return questions[:num_questions]

    def _fallback_text_questions(self, profile: dict, content: str, num_questions: int) -> list:
        """
        Generate universal questions for ANY text document when LLM is not available.
        These questions work for any type of document regardless of domain.
        """
        print(f"📃 Generating fallback text document questions...")
        
        # Get domain information if available
        domain = "General"
        if 'domain' in profile:
            domain = profile.get('domain')
        elif content:
            # Try to infer domain from content
            content_lower = content.lower()
            if any(term in content_lower for term in ['research', 'study', 'methodology', 'findings', 'results']):  
                domain = "Research/Academic"
            elif any(term in content_lower for term in ['revenue', 'market', 'business', 'profit', 'financial']):  
                domain = "Business/Finance"
            elif any(term in content_lower for term in ['patient', 'treatment', 'medical', 'health', 'clinical']):  
                domain = "Healthcare/Medical"
            elif any(term in content_lower for term in ['algorithm', 'code', 'software', 'data', 'programming']):  
                domain = "Technology/Software"
        
        print(f"🌐 Using domain: {domain} for fallback questions")
        
        # Universal question templates that work for ANY document
        universal_templates = [
            "What are the main topics discussed in this document?",
            "What are the key concepts or ideas presented?",
            "What is the primary purpose of this document?",
            "What insights can be drawn from this text?",
            "What conclusions or findings are presented?",
            "What recommendations or suggestions are provided?",
            "What problems or challenges are identified?",
            "What solutions or approaches are mentioned?",
            "What evidence or examples support the main points?",
            "What are the practical applications of the content?",
            "Who is the intended audience for this document?",
            "What background knowledge is assumed?",
            "How is the information organized or structured?",
            "What are the key takeaways from this document?",
            "What future directions or implications are discussed?",
            "What limitations or considerations are mentioned?",
            "What methodology or approach is described?",
            "What trends or patterns are identified?",
            "What terminology or concepts are defined?",
            "What references or sources are cited?"
        ]
        
        # Domain-specific questions to enhance relevance
        domain_questions = {
            "Research/Academic": [
                "What research methodology is used in this study?",
                "What are the key findings and how significant are they?",
                "How does this research contribute to the existing literature?",
                "What limitations are acknowledged in the research design?",
                "What future research directions are suggested?"
            ],
            "Business/Finance": [
                "What business strategies or models are discussed in this document?",
                "What market trends or economic factors are analyzed?",
                "What financial performance indicators are highlighted?",
                "What competitive advantages or challenges are identified?",
                "What recommendations are made for business improvement?"
            ],
            "Healthcare/Medical": [
                "What medical conditions or treatments are discussed in this document?",
                "What health outcomes or patient benefits are described?",
                "What evidence supports the clinical approaches mentioned?",
                "What healthcare guidelines or best practices are referenced?",
                "What patient populations or demographics are specifically addressed?"
            ],
            "Technology/Software": [
                "What technical solutions or architectures are described?",
                "What algorithms or data processing methods are explained?",
                "What system requirements or specifications are outlined?",
                "What development challenges or technical limitations are mentioned?",
                "What user experience considerations or interface designs are discussed?"
            ]
        }
        
        # Select questions based on document characteristics
        questions = []
        
        # Add domain-specific questions first if applicable
        domain_specific = domain_questions.get(domain, [])
        for question in domain_specific:
            if len(questions) < num_questions:
                questions.append(question)
        
        # Add priority universal questions
        priority_questions = [
            "What are the main topics discussed in this document?",
            "What are the key concepts or ideas presented?",
            "What is the primary purpose of this document?",
            "What insights can be drawn from this text?",
            "What conclusions or findings are presented?"
        ]
        
        for question in priority_questions:
            if len(questions) < num_questions and question not in questions:  # Avoid duplicates
                questions.append(question)
        
        # Add remaining universal questions
        for template in universal_templates:
            if len(questions) >= num_questions:
                break
            if template not in questions:  # Avoid duplicates
                questions.append(template)
        
        print(f"✅ Generated {len(questions)} fallback text questions")
        return questions[:num_questions]

    def _fallback_questions(self, profile: dict, num_questions: int) -> list:
        """Generate a small set of generic, dataset-agnostic questions plus column-based templates."""
        questions = []
        columns = profile.get('columns', [])

        generic = [
            "What are the top 5 most frequent categories across key categorical columns?",
            "Which variables show the strongest correlations with the main target or outcome variable (if any)?",
            "What are the notable outliers and how do they impact summary statistics?",
            "How do key metrics differ across major segments (e.g., by a categorical column)?",
            "What time-based trends or seasonality are visible in any date/time columns?",
            "Which fields have the highest missingness and how might that affect analysis?",
            "What combinations of variables are most predictive of important outcomes?",
            "Which entities (e.g., categories, users, items) contribute most to negative vs positive outcomes?",
            "What distributions (skew/kurtosis) suggest data transformations might be needed?",
            "Which comparisons (top/bottom) reveal the largest performance gaps?",
        ]
        questions.extend(generic[:num_questions])

        # If still need more, add column-specific templates
        if len(questions) < num_questions and columns:
            column_templates = [
                "What is the distribution of values in {}?",
                "How does {} correlate with other key variables?",
                "Are there any outliers or unusual patterns in {}?",
                "How does the average of {} vary across a key categorical column?",
            ]
            for col in columns:
                if len(questions) >= num_questions:
                    break
                col_name = col.get('name', str(col)) if isinstance(col, dict) else str(col)
                template = column_templates[(len(questions)) % len(column_templates)]
                questions.append(template.format(col_name))

        return questions[:num_questions]


# ===================== Stateful Wrapper =====================
def question_generation_node(dataset_name: str, df_profile: dict, num_questions: int = 10):
    """Generate understanding + questions for dataset and update WorkflowState."""
    print(f"🔍 QuestionGen - Starting for dataset: {dataset_name}")
    
    # Get data understanding
    understanding = data_understanding(df_profile)
    print(f"🔍 QuestionGen - Understanding: {understanding.get('domain', 'Unknown')}")

    # Generate questions
    agent = QuestionGenerationAgent()
    questions = agent.generate(df_profile, understanding, num_questions)
    
    print(f"🔍 QuestionGen - Agent returned {len(questions)} questions")
    print(f"🔍 QuestionGen - Questions type: {type(questions)}")
    print(f"🔍 QuestionGen - First question: {questions[0] if questions else 'NONE'}")

    # Update global state
    STATE.understanding[dataset_name] = understanding
    STATE.questions[dataset_name] = questions
    
    # Verify the update
    stored_questions = STATE.questions.get(dataset_name, [])
    print(f"🔍 QuestionGen - Stored in STATE: {len(stored_questions)} questions")

    print(f"\n✅ Generated Questions for '{dataset_name}':")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    return STATE
