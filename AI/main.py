from functools import lru_cache, wraps
from flask import Flask, request, jsonify
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from transformers import BertTokenizer, BertModel
import torch
from tavily import TavilyClient
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import spacy

from domain import LEARNING_PATHS, TECH_CATEGORIES, get_resources_for_topic


app = Flask(__name__)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', attn_implementation='eager')


# Load SpaCy model for better text processing
nlp = spacy.load('en_core_web_sm')

with open('/home/Oviemena/Learning_Path_Generator/AI/classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
    clf = model_data['classifier']
    vectorizer = model_data['vectorizer']


api_key = os.getenv("TAVILY_API_KEY",  "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")
tavily = TavilyClient(api_key=api_key)

# Add after API key setup
try:
    # Verify API key on startup
    test_search = tavily.search(query="test", max_results=1)
    print("✅ Tavily API connection successful")
except Exception as e:
    print(f"⚠️  Warning: Tavily API initialization failed: {str(e)}")
    print("➡️  Using fallback resources")
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(
        f"Attempt {retry_state.attempt_number} failed. Retrying in 1 second..."
    )
)
    
@lru_cache(maxsize=100)
def _extract_domain_and_topics(text):
    """Extract domain and topics using BERT embeddings and TECH_CATEGORIES"""
    from domain import TECH_CATEGORIES
    
    doc = nlp(text.lower())
    
    # Direct keyword matching first
    text_tokens = set(doc.text.split())
    for domain, keywords in TECH_CATEGORIES['keywords'].items():
        if any(kw in text_tokens for kw in keywords):
            # Found direct match, use this domain
            domain_keywords = set(TECH_CATEGORIES['keywords'][domain])
            topics = [
                token.text for token in doc 
                if not token.is_stop and token.is_alpha and 
                (token.text in domain_keywords or any(kw in token.text for kw in domain_keywords))
            ]
            return domain, topics[:3] if topics else [token.text for token in doc if not token.is_stop][:3]
    
    # Get text embedding
    text_embedding = model(**tokenizer(text, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
    
    # Find most relevant domain using tech categories
    max_similarity = 0
    chosen_domain = 'general'
    
    for category, keywords in TECH_CATEGORIES['keywords'].items():
        # Create category embedding from keywords
        category_text = ' '.join(keywords)
        category_embedding = model(**tokenizer(category_text, return_tensors="pt")).last_hidden_state.mean(dim=1)
        
        # Convert PyTorch tensors to NumPy arrays for sklearn's cosine_similarity
        text_embedding_numpy = text_embedding.detach().cpu().numpy()
        category_embedding_numpy = category_embedding.detach().cpu().numpy()
        
        similarity = cosine_similarity(
            text_embedding_numpy, 
            category_embedding_numpy
        )[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            chosen_domain = category
    
    # Extract relevant topics
    # Remove common words and keep domain-specific terms
    domain_keywords = set(TECH_CATEGORIES['keywords'].get(chosen_domain, []))
    topics = []
    
    for token in doc:
        if (not token.is_stop and 
            token.is_alpha and 
            (token.text in domain_keywords or 
             any(kw in token.text for kw in domain_keywords))):
            topics.append(token.text)
    
    # If no specific topics found, use main terms from input
    if not topics:
        topics = [token.text for token in doc if not token.is_stop and token.is_alpha]
    
    return chosen_domain, topics[:3]

def _generate_step_description(resources, level, topic_str):
    """Generate meaningful step description based on resource content analysis"""
    try:
        # Collect content summaries from resources
        content_summaries = []
        for resource in resources:
            try:
                # Use Tavily to get content summary
                search_result = tavily.search(
                    query=f"what is {resource['title']} about? summary",
                    max_results=1,
                    search_depth="moderate"
                )
                if search_result.get('results'):
                    content_summaries.append(search_result['results'][0].get('content', ''))
            except Exception as e:
                print(f"Content analysis error: {str(e)}")
                continue

        # Extract key concepts from content summaries
        all_content = ' '.join(content_summaries)
        doc = nlp(all_content)
        
        # Extract important noun phrases and verbs
        key_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                key_phrases.append(chunk.text.lower())
        
        # Filter out topic terms and common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'to', 'for', 'in', 'on', 'at',
                       'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
        topic_terms = set(topic_str.lower().split())
        
        key_phrases = [phrase for phrase in key_phrases 
                      if not any(word in common_words for word in phrase.split())
                      and not any(term in phrase for term in topic_terms)]
        
        # Generate level-appropriate description
        if key_phrases:
            if level == 'beginner':
                concepts = ', '.join(key_phrases[:3])
                return f"Learn {topic_str} fundamentals including {concepts}"
            elif level == 'intermediate':
                concepts = ', '.join(key_phrases[:3])
                return f"Build {topic_str} applications focusing on {concepts}"
            else:
                concepts = ', '.join(key_phrases[:3])
                return f"Master advanced {topic_str} techniques with emphasis on {concepts}"
        
        # Fallback to templates if no meaningful content found
        template = LEARNING_PATHS[level]['templates'][0]
        return template.format(topic=topic_str)
        
    except Exception as e:
        print(f"Description generation error: {str(e)}")
        template = LEARNING_PATHS[level]['templates'][0]
        return template.format(topic=topic_str)

def _predict_level(text):
    # Vectorize input and predict level
    vec = vectorizer.transform([text])
    level = clf.predict(vec)[0]
    return level



def _generate_learning_path(domain, topics, level):
    topic_str = ' '.join(topics)
    steps = []
    seen_descriptions = set()
    
    # Generate steps for all levels
    for current_level in ['beginner', 'intermediate', 'advanced']:
        # Get resources first
        resources = _scrape_resources(topic_str, domain, current_level)
        
        # Generate description based on resources
        description = _generate_step_description(resources, current_level, topic_str)
        
        if description in seen_descriptions:
            continue
            
        seen_descriptions.add(description)
        
        steps.append({
            'description': description,
            'level': current_level,
            'resources': resources
        })
    
    return steps
 
def _get_tech_resources(description, domain, level):
    """Dynamically generate technology-specific resource URLs"""
    # Extract main technology name
    tech_terms = description.lower().split()
    main_tech = next((term for term in tech_terms if term in TECH_CATEGORIES['keywords'].get(domain, [])), '')
    
    if not main_tech:
        return None
        
    # Generate dynamic resource URLs
    tech_resources = {
        'documentation': [
            f'https://docs.{main_tech}.dev',
            f'https://{main_tech}.dev',
            f'https://{main_tech}.org'
        ],
        'learning': [
            f'https://learn.{main_tech}.dev',
            f'https://tutorial.{main_tech}.com'
        ]
    }
    
    return tech_resources

def _validate_resource_url(url):
    """Validate and format resource URLs"""
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    # Remove any trailing slashes
    url = url.rstrip('/')
    
    # Add www for common domains if missing
    common_domains = ['github.com', 'youtube.com', 'medium.com']
    for domain in common_domains:
        if domain in url and not url.startswith(f'https://www.'):
            url = url.replace('https://', 'https://www.')
    
    return url


def _scrape_resources(description, domain, level):
    """Scrape resources with proper level handling and tech-specific resources"""
    # Default fallback resources for each level
    fallback_resources = {
        'beginner': [
            {'title': 'FreeCodeCamp Tutorials', 'link': 'https://www.freecodecamp.org'},
            {'title': 'W3Schools Tutorials', 'link': 'https://www.w3schools.com'},
            {'title': 'MDN Web Docs', 'link': 'https://developer.mozilla.org'}
        ],
        'intermediate': [
            {'title': 'Medium Programming Articles', 'link': 'https://medium.com/topic/programming'},
            {'title': 'Dev.to Community', 'link': 'https://dev.to'},
            {'title': 'GitHub Learning Lab', 'link': 'https://lab.github.com'}
        ],
        'advanced': [
            {'title': 'GitHub Advanced Topics', 'link': 'https://github.com/topics'},
            {'title': 'Stack Overflow', 'link': 'https://stackoverflow.com'},
            {'title': 'ArXiv CS Papers', 'link': 'https://arxiv.org/list/cs/recent'}
        ]
    }
    
    try: # Get search configuration
        # Get tech-specific resources first
        tech_resources = _get_tech_resources(description, domain, level)
        resource_domains = get_resources_for_topic(description, level)

        # Add tech-specific domains if available
        if tech_resources:
            for resource_type, urls in tech_resources.items():
                resource_domains.extend([url.replace('https://', '') for url in urls])

        # Level-specific search terms
        level_terms = {
            'beginner': 'tutorial OR basics OR "getting started"',
            'intermediate': 'intermediate OR "best practices" OR examples',
            'advanced': 'advanced OR expert OR optimization'
        }

        # Specific searches for different content types
        searches = [
            # Video tutorials
            {
                'query': f"{description} {level} {level_terms[level]} video tutorial",
                'domains': ['youtube.com'],
                'max_results': 1
            },
            # Documentation and learning resources
            {
                'query': f"{description} {level} {level_terms[level]} documentation",
                'domains': [d for d in resource_domains if 'youtube.com' not in d],
                'max_results': 2
            }
        ]

        all_results = []
        for search_config in searches:
            try:
                search_results = tavily.search(
                    query=search_config['query'],
                    max_results=search_config['max_results'],
                    search_depth="basic",
                    include_domains=search_config['domains']
                )

                results = [
                {
                    'title': result['title'],
                    'link': _validate_resource_url(result['url'])
                }
                for result in search_results.get('results', [])
                if _is_relevant_resource(result['title'], description, level)
            ]


                all_results.extend(results)

            except Exception as e:
                print(f"Search error for {search_config['query']}: {str(e)}")
                continue

        # Only fallback if no results found
        if all_results:
            return all_results
        
        # Use tech-specific resources as first fallback if available
        if tech_resources:
            tech_specific_results = []
            for resource_type, urls in tech_resources.items():
                for url in urls:
                    tech_specific_results.append({
                        'title': f"{description.title()} {resource_type.title()}",
                        'link': url
                    })
            return tech_specific_results

        # Use general fallback as last resort
        return fallback_resources[level]

    except Exception as e:
        print(f"Resource fetching error: {str(e)}")
        # return fallback_resources[level]
        raise

    
def _is_relevant_resource(title, description, level):
    title_lower = title.lower()
    
    # Keywords that indicate difficulty level
    level_indicators = {
        'beginner': ['beginner', 'basic', 'start', 'introduction', 'fundamental'],
        'intermediate': ['intermediate', 'practical', 'improve'],
        'advanced': ['advanced', 'expert', 'master', 'professional']
    }
     # Check if title contains indicators of a different level
    for check_level, indicators in level_indicators.items():
        if check_level != level and any(indicator in title_lower for indicator in indicators):
            return False
    
    # Check relevance using simple heuristics
    topic_terms = set(description.lower().split())
    title_terms = set(title.split())
    
    relevance_score = len(topic_terms.intersection(title_terms))
    level_match = level.lower() in title.lower()
    
    return relevance_score > 0 or level_match
    
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text'].lower()
    
    # Extract domain and topics
    domain, topics = _extract_domain_and_topics(text)
    
    # Predict level
    level = _predict_level(text)
    
    # Generate steps
    steps = _generate_learning_path(domain, topics, level)
    
    # Add resources
    for step in steps:
        step['resources'] = _scrape_resources(step['description'], domain, step['level'])

    return jsonify({'steps': steps})

if __name__ == '__main__':
    app.run(port=5001)