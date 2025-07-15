# MEDSCAN-AI
MedScan AI
MedScan AI is an advanced, multi-faceted medical diagnostic platform developed to assist healthcare professionals and researchers in the detection, classification, and interactive understanding of various diseases using state-of-the-art machine learning and artificial intelligence.

Features
1. Pneumonia Detection via Audio Classification
Utilizes audio recordings (such as lung sounds) to distinguish between healthy individuals and patients with pneumonia.

Employs audio processing techniques and machine learning for accurate classification.

2. Heart Disease Classification
Uses patient data (potentially ECG, EHR features, biomarkers) for risk analysis and prediction.

Implements a Random Forest Classifier for robust and explainable heart disease categorization.

3. Wilson Disease Diagnosis
Analyzes clinical and biomedical data to predict the presence of Wilson Disease.

Random Forest Classifier ensures reliable performance on diverse datasets.

4. Brain Tumor Detection
Integrates medical imaging (MRI, CT scans) for the automated detection and classification of brain tumors.

Leverages deep neural networks for high accuracy in tumor identification.

5. Interactive Medical Question Answering
Features a BERT-powered natural language assistant for answering user queries.

Retrieves information from a stored medical glossary file, offering clear and concise explanations for medical terms and concepts.

Project Architecture
Module	Description	Algorithm/Model
Pneumonia Detection	Classifies lung audio for pneumonia	Audio Processing + ML
Heart Disease	Predicts heart disease from patient data	Random Forest Classifier
Wilson Disease	Diagnoses Wilson Disease	Random Forest Classifier
Brain Tumor Detection	Identifies tumors from brain scans	Neural Network
Medical QA Assistant	Answers health-related queries from glossary	BERT Model
Usage
Clone the Repository:

bash
git clone https://github.com/Ansh-ML-wq/MEDSCAN-AI.git
cd MEDSCAN-AI
Installation:

Ensure Python 3.7+ is installed.

Install required packages:

bash
pip install -r requirements.txt
Running Modules:

Each disease detection module is standalone and can be executed via its respective script.

The BERT-based Q&A assistant launches as a command-line or web interface for interaction.

Input Data:

Provide data in the specified format (.wav for audio, .csv for tabular data, medical images for brain tumors).

Example Workflow
Detect Pneumonia: Upload lung audio to /audio_detection.ipynb/ and run it.

Classify Heart Disease: Place patient CSV in /HDD.ipynb/, then run it.

Brain Tumor Detection: Upload MRI images, run predict_brain_tumor.py for automated report.

Interact with Medical Assistant: Launch ii.ipynb to ask medical questions and receive answers from the glossary.
and finally run the nav.py using streamllit run nav.py which is the integrated usear interface frontend

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

For major changes, please discuss them via issue before implementation.
