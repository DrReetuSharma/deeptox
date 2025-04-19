

Key Features:
=======
## AI-Tox: AI-Powered Toxicity Mapping
### AI-Tox is an AI-powered tool designed to predict the toxicity of chemical compounds based on their SMILES (Simplified Molecular Input Line Entry System) representations. By leveraging machine learning models like ChemBERTa, this tool helps researchers and scientists assess the potential toxic effects of molecules, aiding in the early stages of drug discovery, environmental safety assessments, and more.

#### Key Features:

Machine Learning-Driven: Uses state-of-the-art AI models (like ChemBERTa and RandomForest) to classify toxicity.

Comprehensive Toxicity Classification: Predicts various toxicity types including hepatotoxicity, nephrotoxicity, and others.

Instant Prediction: Provides real-time toxicity predictions.

User-Friendly Interface: Simple interface for inputting SMILES strings of molecules.

Downloadable Results: Download toxicity predictions in a tabular format for further analysis.


How It Works:
=======
#### How It Works:

Input: Users input a molecule's SMILES string into the provided input field.

Processing: The app processes the SMILES string through a pre-trained machine learning model.

Output: The app returns the predicted toxicity class, such as Hepatotoxicity, Genotoxicity, and more, along with confidence scores.


Requirements:
=======
##### Requirements:

Python 3.10 or higher

Gradio for the interactive web interface

Transformers for using pre-trained machine learning models

Torch for deep learning support

Pandas, NumPy, scikit-learn for data manipulation and analysis

<<<<<<< HEAD
Gunicorn for running the app in production
=======
Gunicorn for running the app in production
>>>>>>> 585a097a50ecd12f9b6e60915adb4271e2978bca
