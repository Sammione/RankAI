from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a lightweight, fast model (runs locally for free)
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_match_score(jd_text, cv_text):
    """
    Calculates the cosine similarity between a Job Description and a CV.
    Returns a score between 0 and 100.
    """
    # 1. Generate Embeddings
    embeddings1 = model.encode(jd_text, convert_to_tensor=True)
    embeddings2 = model.encode(cv_text, convert_to_tensor=True)

    # 2. Compute Cosine Similarity
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    # 3. Convert to percentage
    score = float(cosine_scores[0][0]) * 100
    return round(score, 2)

def rank_cvs(jd_text, cv_list):
    """
    Ranks a list of CVs based on a Job Description.
    cv_list should be a list of dictionaries: [{'id': 1, 'text': '...'}, ...]
    """
    ranked_results = []
    
    for cv in cv_list:
        score = calculate_match_score(jd_text, cv['text'])
        ranked_results.append({
            "applicant_id": cv.get('id'),
            "name": cv.get('name', 'Unknown'),
            "score": score,
            "match_level": "High" if score > 70 else "Medium" if score > 40 else "Low"
        })
    
    # Sort by score descending
    return sorted(ranked_results, key=lambda x: x['score'], reverse=True)
