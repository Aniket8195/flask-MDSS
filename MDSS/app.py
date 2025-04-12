from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load pre-trained models and preprocessors
model_1 = joblib.load("a_random_forest_model.pkl")
rf_model_2 = joblib.load("random_forest_model_2.pkl")
num_imputer = joblib.load("a_num_imputer.pkl")
cat_imputer = joblib.load("a_cat_imputer.pkl")
encoder = joblib.load("a_encoder.pkl")
scaler = joblib.load("a_scaler.pkl")
tfidf_vectorizer = joblib.load("a_tfidf_vectorizer.pkl")

# Define column names
numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'anchor_age', 'prev_visit', 'prev_adm']
categorical_cols = ['insurance', 'language', 'marital_status', 'ethnicity', 'gender']

# Load BERT model for embeddings
model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)
model_2_bert = AutoModel.from_pretrained(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_2_bert = model_2_bert.to(device)

def flatten_embeddings(embeddings):
    return embeddings.flatten() if isinstance(embeddings, np.ndarray) else np.zeros(embeddings.shape[0] * embeddings.shape[1])

def get_embeddings_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls_embeddings

def get_model_1_prediction(df):
    df[numerical_cols] = num_imputer.transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.transform(df[categorical_cols])
    
    # TF-IDF for 'chiefcomplaint'
    chiefcomplaint_tfidf = tfidf_vectorizer.transform(df['chiefcomplaint'].fillna('')).toarray()
    chiefcomplaint_tfidf_df = pd.DataFrame(chiefcomplaint_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    df.drop(columns=['chiefcomplaint'], inplace=True)
    df = pd.concat([df, chiefcomplaint_tfidf_df], axis=1)
    
    # One-hot encoding
    categorical_encoded = encoder.transform(df[categorical_cols])
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, categorical_encoded_df], axis=1)
    
    # Scale numerical columns
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    df.columns = df.columns.astype(str)
    
    prediction = model_1.predict(df)
    #probability_1 = model_1.predict_proba(df)
    
    return prediction

def get_model_2_prediction(df_2):
    medications = df_2['medications'].tolist()
    diagnoses = df_2['diagnosis'].tolist()
    
    medication_embeddings = get_embeddings_batch(medications, tokenizer, model_2_bert, device)
    diagnosis_embeddings = get_embeddings_batch(diagnoses, tokenizer, model_2_bert, device)
    
    df_2['medication_embeddings_flat'] = [flatten_embeddings(embed) for embed in medication_embeddings]
    df_2['diagnosis_embeddings_flat'] = [flatten_embeddings(embed) for embed in diagnosis_embeddings]
    
    medication_flat = pd.DataFrame(df_2['medication_embeddings_flat'].tolist())
    diagnosis_flat = pd.DataFrame(df_2['diagnosis_embeddings_flat'].tolist())
    
    features_2 = pd.concat([df_2[['gender', 'age', 'Troponin T(ng/ML)', 'emergency']], medication_flat, diagnosis_flat], axis=1)
    features_2.columns = features_2.columns.astype(str)
    
    prediction = rf_model_2.predict(features_2)
    return prediction

def fusion_function(prediction_1, prediction_2):
  
    weight_1=1
    weight_2=1


    if isinstance(prediction_1, (float, np.float64)) and isinstance(prediction_2, (float, np.float64)):
        combined_score = (weight_1 * prediction_1 + weight_2 * prediction_2) / (weight_1 + weight_2)
        # Apply a threshold of 0.5 to convert back to binary output
        return 1 if combined_score >= 0.5 else 0
    
    # If the predictions are binary, simply return the weighted average (integer)
    else:
        combined_prediction = (weight_1 * prediction_1 + weight_2 * prediction_2) / (weight_1 + weight_2)
        return np.round(combined_prediction).astype(int)  # Round to 0 or 1

@app.route('/predict_fusion', methods=['POST'])
def predict_fusion():
    try:
        data = request.get_json()
        print(data)
        # Convert input data to DataFrame
        df_1 = pd.DataFrame([{
            'temperature': data.get('temperature', np.nan),
            'heartrate': data.get('heartrate', np.nan),
            'resprate': data.get('resprate', np.nan),
            'o2sat': data.get('o2sat', np.nan),
            'sbp': data.get('sbp', np.nan),
            'dbp': data.get('dbp', np.nan),
            'pain': data.get('pain', np.nan),
            'acuity': data.get('acuity', np.nan),
            'anchor_age': data.get('anchor_age', np.nan),
            'prev_visit': data.get('prev_visit', np.nan),
            'prev_adm': data.get('prev_adm', np.nan),
            'insurance': data.get('insurance', None),
            'language': data.get('language', None),
            'marital_status': data.get('marital_status', None),
            'ethnicity': data.get('ethnicity', None),
            'gender': data.get('gender', None),
            'chiefcomplaint': data.get('chiefcomplaint', '')
        }])
        
        df_2 = pd.DataFrame([{
            'gender': data.get('gender', None),
            'age': data.get('age', None),
            'Troponin T(ng/ML)': data.get('troponin', np.nan),
            'emergency': data.get('emergency', None),
            'medications': data.get('medications', ''),
            'diagnosis': data.get('diagnosis', '')
        }])

        # Ensure required columns exist
        if df_1.isnull().all().all() or df_2.isnull().all().all():
            return jsonify({'error': 'Invalid input. Provide required fields for both models.'}), 400
        
        # Get predictions from both models
        prediction_1 = get_model_1_prediction(df_1)
        prediction_2 = get_model_2_prediction(df_2)
        
        # Compute the fusion output
        fused_prediction = fusion_function(prediction_1, prediction_2)

        # Convert the results to native Python types for JSON serialization
          # Convert probabilities to float
        prediction_1 = int(prediction_1)  # Convert to int if necessary
        prediction_2 = int(prediction_2)  # Convert to int if necessary
        fused_prediction = int(fused_prediction)  # Convert to int if necessary

        return jsonify({
            'model_1_prediction': prediction_1,
            'model_2_prediction': prediction_2,
            'fused_prediction': fused_prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)