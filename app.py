import gradio as gr
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import shutil
import uuid

# Load model and tokenizer
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=12)
model.eval()

toxicity_labels = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-γ", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

toxicity_levels = {
    "NR-AR": [0.3, 0.6], "NR-AR-LBD": [0.2, 0.5], "NR-AhR": [0.25, 0.55], "NR-Aromatase": [0.35, 0.7],
    "NR-ER": [0.3, 0.6], "NR-ER-LBD": [0.4, 0.75], "NR-PPAR-γ": [0.2, 0.5], "SR-ARE": [0.3, 0.6],
    "SR-ATAD5": [0.2, 0.45], "SR-HSE": [0.25, 0.5], "SR-MMP": [0.3, 0.65], "SR-p53": [0.35, 0.7]
}

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def score_to_level(score, label):
    low_threshold, high_threshold = toxicity_levels[label]
    if score < low_threshold:
        return "L"  # Low
    elif score <= high_threshold:
        return "M"  # Medium
    else:
        return "H"  # High

def predict_toxicity(file):
    unique_id = uuid.uuid4().hex
    filename = f"toxicity_results_{unique_id}.csv"

    filepath = shutil.copy(file.name, os.path.join(UPLOAD_DIR, os.path.basename(file.name)))
    df = pd.read_csv(filepath)
    results = []

    for _, row in df.iterrows():
        smiles_id = row[0]
        smiles = row[1]

        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        result_row = {"ID": smiles_id, "SMILES": smiles}
        result_row.update({tox: round(p, 3) for tox, p in zip(toxicity_labels, probs)})
        result_row.update({f"{tox}_level": score_to_level(p, tox) for tox, p in zip(toxicity_labels, probs)})
        results.append(result_row)

    result_df = pd.DataFrame(results)
    output_path = os.path.join(RESULT_DIR, filename)
    result_df.to_csv(output_path, index=False)

    return result_df, output_path

# Interface with downloadable link and updated details
def app_with_download(file):
    df, download_path = predict_toxicity(file)
    return df, download_path

iface = gr.Interface(
    fn=app_with_download,
    inputs=gr.File(label="Upload input_smiles.csv (ID,SMILES format)"),
    outputs=[gr.Dataframe(label="Toxicity Predictions"), gr.File(label="Download CSV")],
    title="DeepToxiLens: AI-Powered Toxicity Level Classifier",
    description="""Version 2 - Enhanced Toxicity Levels Classification with M, L, H Scoring.
    This app predicts the toxicity levels (Low, Medium, High) for 12 different toxicity endpoints based on molecular structures.
    Each toxicity endpoint is classified into Low (L), Medium (M), or High (H) depending on the predicted probability.
    Please upload a CSV with columns 'ID' and 'SMILES' for prediction.""",
    theme="compact"
)

iface.launch(share=True)
