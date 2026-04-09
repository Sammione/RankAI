from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a lightweight, fast model (runs locally for free)
model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_cvs(jd_text, cv_list):
    """
    Ranks a list of CVs based on a Job Description.
    cv_list should be a list of dictionaries: [{'id': 1, 'text': '...'}, ...]
    """
    # 1. Generate Embedding for Job Description ONCE
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    
    ranked_results = []
    
    for cv in cv_list:
        # 2. Generate Embedding for CV
        cv_embedding = model.encode(cv.get('text', ''), convert_to_tensor=True)
        
        # 3. Compute Cosine Similarity
        cosine_score = util.cos_sim(jd_embedding, cv_embedding)
        score = float(cosine_score[0][0]) * 100
        score = round(score, 2)

        ranked_results.append({
            "applicant_id": cv.get('id'),
            "name": cv.get('name', 'Unknown'),
            "score": score,
            "match_level": "High" if score > 70 else "Medium" if score > 40 else "Low"
        })
    
    # Sort by score descending
    return sorted(ranked_results, key=lambda x: x['score'], reverse=True)
