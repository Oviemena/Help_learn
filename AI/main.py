from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from tavily import TavilyClient
from sklearn.cluster import KMeans
import numpy as np
import pickle
import os


app = Flask(__name__)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


with open('AI/classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
    clf = model_data['classifier']
    vectorizer = model_data['vectorizer']


api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key)

def _extract_topics(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    valid_indices = [i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]
    valid_embeddings = embeddings[valid_indices]
    valid_tokens = [tokens[i] for i in valid_indices]
    
    if len(valid_embeddings) > 1:
        kmeans = KMeans(n_clusters=min(2, len(valid_embeddings)), random_state=42)
        labels = kmeans.fit_predict(valid_embeddings)
        
        clusters = {}
        for token, label in zip(valid_tokens, labels):
            clusters.setdefault(label, []).append(token)
        topics = [' '.join(cluster) for cluster in clusters.values()]
    else:
        topics = valid_tokens or ['general']

    return topics

def _predict_level(text):
    # Vectorize input and predict level
    vec = vectorizer.transform([text])
    level = clf.predict(vec)[0]
    return level


def _generate_steps(text, topics):
    # Predict base level
    base_level = _predict_level(text)
    levels = ['beginner', 'intermediate', 'advanced']
    start_idx = levels.index(base_level)
    
    # Generate steps based on level
    steps = []
    primary_topic = topics[0]
    
    # Add steps dynamically
    for i in range(start_idx, min(start_idx + 2, 3)):  # Up to 2 steps from base level
        level = levels[i]
        if i == start_idx:
            desc = f"Start with {primary_topic} at {level} level"
        else:
            secondary = topics[1] if len(topics) > 1 else 'practical applications'
            desc = f"Advance to {primary_topic} with {secondary} ({level})"
        steps.append({'description': desc, 'level': level})
    
    return steps


def _scrape_resources(description, level):
    params = {
        "q": f"{description} {level} tutorial site:*.edu|site:*.org|site:youtube.com -inurl:(signup | login)",
    }
    search = tavily.search(query=params, max_results=3)
    resources = [
        {'title': result.get('title', 'No title'), 'link': result.get('url', 'https://tavily.com')}
        for result in search.get('results', [])
    ] 
    return resources if resources else [{'title': 'No resources found', 'link': 'https://tavily.com'}]


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text'].lower()
    
    topics = _extract_topics(text)
    steps = _generate_steps(text, topics)
    
    for step in steps:
        step['resources'] = _scrape_resources(step['description'], step['level'])

    return jsonify({'steps': steps})

if __name__ == '__main__':
    app.run(port=5001)