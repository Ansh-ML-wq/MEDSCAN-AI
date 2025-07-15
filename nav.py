import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model 
from scipy.io.wavfile import write
import os
import librosa
import numpy as np
import joblib
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pypdf
import time
# Set the page configuration
st.set_page_config(page_title="MedScan AI", layout="wide")

# Custom CSS for white background
st.markdown(
    """
    <style>
    /* Set the main background color to white */
    .reportview-container {
        background: white;
    }
    /* Set the sidebar background to white as well */
    .sidebar .sidebar-content {
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the webpage content
#st.title("MEDSCAN AI")
#st.subheader("Diseases and Conditions")

# Sidebar content
st.sidebar.title("Disease Checker")
with st.sidebar:
    selected = option_menu(" ",["Home","Pneumonia Detection Using audio","Brain Tumor Detection",
                            "Heart Disease Detection","Wilson Disease Detection"],menu_icon="cast",default_index=0)
if selected == "Home":
    st.title("MEDSCAN AI")
    groq_api_key = st.secrets["groq_api_key"]
    google_api_key = st.secrets["google_api_key"]
    st.title("MEDICAl Q&A")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192"
        )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """
        )
    def vector_embedding():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
                )
            st.session_state.loader = PyPDFLoader(r"C:\Users\ansh1\OneDrive\Desktop\Dictionary of Medical Terms.pdf")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    prompt1 = st.text_input("Enter Your Question")
    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        st.write(response['answer'])
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")


elif selected == "Pneumonia Detection Using audio":
    gru_model = load_model('RD.h5')
    def preprocess_audio(data, sampling_rate):
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=52).T, axis=0)
        return mfccs
    def classify_audio(data, sampling_rate):
        mfccs = preprocess_audio(data, sampling_rate)
        mfccs_reshaped = np.reshape(mfccs, (1, 1, 52))
        prediction = gru_model.predict(mfccs_reshaped)
        classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        return predicted_class
    st.title("Pneumonia Audio Detection")
    option = st.selectbox("Select an option", ["Upload Audio File", "Record Voice"])
    if option == "Upload Audio File":
        uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    if uploaded_file is not None:
        data, sampling_rate = librosa.load(uploaded_file)
        predicted_class = classify_audio(data, sampling_rate)
        if st.button("Classify"):
            if predicted_class == "Healthy":
                st.success("Healthy")
            elif predicted_class == "COPD":
                st.warning("COPD")
                st.title("Chronic Obstructive Pulmonary Disease (COPD)")
                st.markdown(""" COPD is a chronic lung disease that obstructs airflow to and from the lungs, 
                            causing breathing difficulties and excessive coughing with mucus production """)
                st.title("Causes and Symptoms")
                st.markdown(""" COPD is often caused by long-term exposure to air pollutants like cigarette smoke, dust, and chemical particulates""")
                st.markdown(""" Symptoms include breathlessness, chronic cough, wheezing, and chest tightness """)
                st.title("Treatment and Prevention")
                st.markdown(""" Treatment involves bronchodilators, steroids, oxygen therapy, and pulmonary rehabilitation""")
                st.markdown(""" Prevention includes quitting smoking, avoiding air pollutants, and managing asthma effectively""")
            elif predicted_class == "Bronchiolitis":
                st.warning("Bronchiolitis")
                st.title("Bronchiolitis")
                st.markdown("""Bronchiectasis: Bronchiectasis occurs when the walls of the bronchi become thickened and damaged, 
                            leading to difficulty in clearing mucus. This damage is typically caused by inflammation and infections,
                             which can be triggered by lung infections or acid reflux """)
                st.title("Symptoms and Causes")
                st.markdown("""Symptoms of bronchiectasis include chronic coughing (sometimes with blood), shortness of breath, fatigue, and chest pain. 
                            Bronchiectasis treatment includes treating lung infections,
                             drinking plenty of fluids, and exercising """)
                st.title("Prevention and Treatement")
                st.markdown("""Bronchiectasis treatment includes treating lung infections, drinking plenty of fluids, and exercising. 
                            To prevent bronchiectasis, seek prompt treatment for lung infections, quit smoking, 
                            and ensure children receive vaccinations for illnesses like whooping cough and measles. """)
            elif predicted_class == "Pneumonia":
                st.warning("Pneumonia")
                st.title("Pneumonia")
                st.markdown("""Pneumonia is an infection of the lungs that can be caused by bacteria, viruses, or fungi """)
                st.title("Symptoms and Causes")
                st.markdown("""Symptoms include cough, fever, chest pain, and difficulty breathing """)
                st.title("Treatment and Prevention")
                st.markdown("""eria, viruses, or fungi. Symptoms include cough, fever, chest pain, and difficulty breathing. Treatment depends on the cause of pneumonia but can include antibiotics for bacterial pneumonia, 
                            antiviral medications for viral pneumonia, and supportive care such as rest and fluids. Prevention includes vaccination against pneumonia and the flu, frequent handwashing, and avoiding smoking. """)
            
            elif predicted_class == "URTI":
                st.warning("URTI")
                st.title("Upper Respiratory Tract Infection (URTI)")
                st.markdown(""" URTIs are infections of the nose and throat, commonly caused by viruses, such as rhinoviruses and coronaviruses """)
                st.title("Symptoms and Causees")
                st.markdown("""Symptoms include sore throat, nasal congestion, runny nose, cough, headache, and fever """)
                st.title("Treatement and Prevention")
                st.markdown("""Treatment focuses on symptom relief with rest, fluids, over-the-counter pain relievers, and decongestants; antibiotics are only effective if the infection is bacterial. Prevention includes frequent handwashing, 
                            avoiding close contact with sick individuals, and getting vaccinated against the flu. """)
                

    
elif selected == "Brain Tumor Detection":
    base_model = load_model("dis.h5")
    classifier = joblib.load('rfc.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    def recognize_face(image):
        image_resized = cv2.resize(image, (224, 224))
        image_array = img_to_array(image_resized) / 255.0
        features = base_model.predict(np.expand_dims(image_array, axis=0))
        prediction = classifier.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label[0]
    st.title("Brain Tumor Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Recognize"):
            result = recognize_face(image)
            if result == "glioma_tumor":
                st.warning("glioma_tumor")
                st.title("glioma_tumor")
                st.markdown("""Gliomas are tumors that arise from glial cells in the brain or spinal cord and can be classified into various types, including astrocytomas, oligodendrogliomas, and ependymomas. These tumors can be benign or malignant, 
                            with malignant gliomas being more aggressive and life-threatening. """)
                st.markdown("""Causes: The exact cause of gliomas remains unclear, but they can occur sporadically or be associated with genetic predispositions. 
                            Factors such as exposure to radiation and certain genetic syndromes may increase the risk """)
                st.title("Symptoms")
                st.markdown("""1) Headaches: Often persistent and worsening over time. """)
                st.markdown("""2) Seizures: Can be one of the first signs, particularly in astrocytomas. """)
                st.markdown("""3) Cognitive Changes: This includes memory loss, confusion, and difficulty speaking """)
                st.markdown("""4) Neurological Deficits: Weakness or numbness in limbs, balance issues, and visual disturbances. """)
                st.title("Treatment")
                st.markdown("""Treatment: Treatment options for gliomas typically involve a combination of:
                            1)Surgery: To remove as much of the tumor as possible.
                            2)Radiation Therapy: Often used post-surgery to target remaining cancerous cells.
                            3)Chemotherapy: May be administered to treat more aggressive forms of glioma.
                            4)Targeted Therapy: Focuses on specific genetic mutations within the tumo """)
                st.title("Prevention and Precautions")
                st.markdown("""Prevention and Precautions: While there is no guaranteed way to prevent gliomas, certain measures may reduce risk:
                            1)Avoiding exposure to known carcinogens, such as radiation.
                            2)Maintaining a healthy lifestyle with a balanced diet and regular exercise.
                            3)Monitoring family history for genetic conditions that could predispose individuals to brain tumors """)
            elif result == "meningioma_tumor":
                st.warning("meningioma_tumor")
                st.title("meningioma_tumor")
                st.markdown("""Meningiomas are tumors that develop from the meninges, the protective membranes covering the brain and spinal cord. They are typically classified into three grades based on their aggressiveness:
                             Grade 1 (benign), Grade 2 (atypical), and Grade 3 (anaplastic). """)
                st.markdown("""Causes: The exact cause of meningiomas is not fully understood, but they may arise from genetic mutations, such as those involving the NF2 gene, and are more common in women. 
                            Risk factors include exposure to radiation and certain genetic syndromes """)
                st.title("Symptoms")
                st.markdown("""Symptoms of meningiomas can vary significantly depending on their location and size but often include:
                            1)Headaches: Commonly reported, particularly if the tumor is large.
                            2)Seizures: May occur as a result of irritation to the brain.
                            3)Neurological Deficits: This can include vision changes, loss of smell or hearing, confusion, and weakness in limbs.
                            4)Cognitive Changes: Memory issues or personality alterations may be observed. """)
                st.title("Treatment")
                st.markdown("""Treatment: Treatment options for gliomas typically involve a combination of:
                            1)Surgery: To remove as much of the tumor as possible.
                            2)Radiation Therapy: Often used post-surgery to target remaining cancerous cells.
                            3)Chemotherapy: May be administered to treat more aggressive forms of glioma.
                            4)Targeted Therapy: Focuses on specific genetic mutations within the tumor """)
                st.title("Prevention and Precautions")
                st.markdown("""Prevention and Precautions: While there is no guaranteed way to prevent gliomas, certain measures may reduce risk:
                            1)Avoiding exposure to known carcinogens, such as radiation.
                            2)Maintaining a healthy lifestyle with a balanced diet and regular exercise.
                            3)Monitoring family history for genetic conditions that could predispose individuals to brain tumors """)
                
            elif result == "pituitary_tumor":
                st.warning("pituitary_tumor")
                st.title("pituitary_tumor")
                st.markdown("""Pituitary tumors are abnormal growths that develop in the pituitary gland, located at the base of the brain. These tumors can be classified as functioning, which secrete hormones, or nonfunctioning, 
                            which do not produce hormones but may still cause symptoms by pressing on surrounding structures. """)
                st.markdown("""Causes: The exact cause of pituitary tumors is not well understood. However, genetic factors, such as mutations in the NF1 or MEN1 genes, may increase susceptibility. 
                            Other potential risk factors include exposure to radiation and certain hormonal conditions """)
                st.title("Symptoms")
                st.markdown("""Symptoms: Symptoms of pituitary tumors can vary widely depending on whether they are functioning or nonfunctioning:
                            1)Functioning Tumors: These tumors produce excess hormones, leading to specific symptoms based on the hormone involved. For example:
                            2)Prolactin-secreting tumors may cause galactorrhea (milky discharge from the nipples), menstrual irregularities in women, and impotence in men.
                            3)Growth hormone-secreting tumors can result in acromegaly (enlarged hands and feet) and joint pain.
                            4)ACTH-secreting tumors may lead to Cushing's syndrome, characterized by weight gain and high blood pressure. """)
                st.markdown("""Nonfunctioning Tumors: These typically present with symptoms due to their size and pressure on nearby structures, including:
                            1)Headaches
                            2)Vision problems (loss of peripheral vision or double vision)
                            3)Facial numbness or pain
                            4)Hormonal deficiencies leading to fatigue, weight changes, and menstrual irregularities. """)
                st.title("Treatment")
                st.markdown("""Treatment: Treatment options for gliomas typically involve a combination of:
                            1)Surgery: To remove as much of the tumor as possible.
                            2)Radiation Therapy: Often used post-surgery to target remaining cancerous cells.
                            3)Chemotherapy: May be administered to treat more aggressive forms of glioma.
                            4)Targeted Therapy: Focuses on specific genetic mutations within the tumor """)
                st.title("Prevention and Precautions")
                st.markdown("""Prevention and Precautions: While there is no guaranteed way to prevent gliomas, certain measures may reduce risk:
                            1)Avoiding exposure to known carcinogens, such as radiation.
                            2)Maintaining a healthy lifestyle with a balanced diet and regular exercise.
                            3)Monitoring family history for genetic conditions that could predispose individuals to brain tumors """)
            
            elif result == "no_tumor":
                st.success("no_tumor")



elif selected == "Heart Disease Detection":
    st.title("Heart Disease Detection")
    st.write("Heart disease, also referred as cardiovascular diseases, is broad term used for diseases and conditions affecting the heart and circulatory system. It is a major cause of disability all around the world. Since heart is amongst the most vital organs of the body, its diseases affect other organs and part of the body as well. There are several different types and forms of heart diseases. The most common ones cause narrowing or blockage of the coronary arteries, malfunctioning in the valves of the heart,"
    "enlargement in the size of heart and several others leading to heart failure and heart attack")
    st.subheader("Key facts according to WHO (World Health Organaizations)")
    st.write("Cardiovascular diseases (CVDs) are the leading cause of death globally."
    "An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke."
    "Over three quarters of CVD deaths take place in low- and middle-income countries."
    "Out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, 38% were caused by CVDs."
    "Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol."
    "It is important to detect cardiovascular disease as early as possible so that management with counselling and medicines can begin.")
    model = pickle.load(open("clf.pkl","rb"))
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=[0,"Female",1,"Male"])
    cp = st.selectbox("Chest Pain Type", options=[0,"Typical Angina",1,"Atypical Angina",2,"Non-Anginal Pain",3,"Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=70)
    chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,"No",1,"Yes"])
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0,"Normal",1,"Having ST-T wave abnormality",2,"Showing probable or definite left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60)
    exang = st.selectbox("Exercise Induced Angina", options=[0,"No",1,"Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest",min_value=0.0,max_value=6.5)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0,"Upsloping",1,"Flat",2,"Downsloping"])
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0)
    thal = st.selectbox("Thalassemia", options=[1,"Normal",2,"Fixed Defect",3,"Reversable Defect"])
    if st.button("Predict"):
        input_data = np.array([[age,
                             sex,
                             cp,
                             trestbps,
                             chol,
                             fbs,
                             restecg,
                             thalach,
                             exang,
                             oldpeak,
                             slope,
                             ca,
                             thal]])
    
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("High Chances of Heart Disease Present")
    else:
        st.success("No Heart Disease")


elif selected == "Wilson Disease Detection":
    st.title("WILSON DISEASE")
    st.markdown("""Wilson disease is a rare genetic condition that occurs when your body accumulates too much copper, especially in the liver and brain. 
    Your body needs a small amount of copper from food to stay healthy, but without treatment,"
    Wilson disease can lead to high copper levels that cause life-threatening organ damage""")
    st.title("Symptoms and Causes")
    st.subheader("What are the Symptoms and Causes ?")
    st.markdown("""Symptoms of Wilson disease vary a lot from person to person.Wilson disease is present at birth (congenital),
                but the symptoms don’t appear until copper builds up in your liver, brain, eyes or other organs. 
                People who have Wilson disease typically develop symptoms between ages 5 and 40. However, 
                some people develop symptoms at younger or older ages.
                Some people are diagnosed with other liver or mental health disorders when they actually have Wilson disease. 
                This is because symptoms can be nonspecific and similar to other conditions until copper levels are measured.
                """)
    col1,col2,col3 = st.columns(3)
    with col1:
        st.title("Liver Symptoms")
        st.markdown("""People with Wilson disease often develop symptoms of hepatitis (inflammation of the liver) and can have an abrupt decrease in liver function (acute liver failure). "
        "These symptoms may include:""")
        st.markdown("""1) Fatigue""")
        st.markdown(""" 2) Nausea and vomiting.""")
        st.markdown("""3) Yellow tint to the whites of your eyes and skin (jaundice). """)
        st.markdown("""4) Swelling of the lower legs, ankles or feet (edema) """)
        st.markdown("""5) Bloating from a buildup of fluid in the abdomen (ascites). """)
    
    with col2:
        st.title("Central nervous system symptoms")
        st.markdown("""People with Wilson disease may develop central nervous system symptoms that affect their mental health as copper builds up in their body. "
        "These symptoms are more common in adults but do also occur in children.""")
        st.markdown("""Nervous system symptoms may include:""")
        st.markdown("""1) Problems with speech, swallowing or physical coordination. """)
        st.markdown("""2) Stiff muscles. """)
        st.markdown("""3) Tremors and Uncontrolled Movements""")
        st.markdown("""4) Changes in mood, personality or behavior. """)
        st.markdown("""5) Anxiety and Depression """)

    with col3:
        st.title("Eye and Other Symptoms")
        st.markdown(""" Many people with Wilson disease have green, gold or brown rings around the edge of the corneas in their eyes (Kayser-Fleischer rings)
                    A buildup of copper in the eyes causes the Kayser-Fleischer rings. 
                    Your healthcare provider can see these rings during a special eye exam called a slit-lamp exam.""")
        st.subheader("Other Symptoms")
        st.markdown("""1) Hemolytic anemia. """)
        st.markdown("""2) Bone and joint problems (arthritis or osteoporosis).""")
        st.markdown("""3) Heart problems (cardiomyopathy).""")
        st.markdown("""4) Kidney problems (renal tubular acidosis or kidney stones).""")

    model = pickle.load(open("clf.pkl","rb"))
    le = LabelEncoder()
    def main():
        st.title("Wilson Disease Detection")
        st.markdown("<p style='color: black;'>Enter body Test Details</p>", unsafe_allow_html=True)
        Age= st.number_input("Age",min_value=3.0000,max_value=80.0000)
        Cl = st.number_input("Ceruloplasmin_Level",min_value=-0.5000,max_value=43.0000)
        cbs = st.number_input("Copper in Blood Serum",min_value=-70.0000,max_value=450.0000)
        Ciu = st.number_input("Copper in Urine",max_value=271.0000,min_value=-17.0000)
        ALT = st.number_input("ALT",min_value=-26.0000,max_value=145.0000)
        AST = st.number_input("AST",min_value=-26.2112,max_value=145.0000)
        TB = st.number_input("Total Bilirubin",min_value=-3.2580,max_value=7.0000)
        GGT = st.number_input("Gamma-Glutamyl Transferase (GGT)",min_value=-6.2580,max_value=187.0000)
        KFR = st.number_input("Kayser-Fleischer Rings",min_value=0.0000,max_value=1.0000)
        NSS = st.number_input("Neurological Symptoms Score",min_value=-3.2580,max_value=14.0000)
        CFS = st.number_input("Cognitive Function Score",min_value=35.2580,max_value=120.0000)
        FH = st.number_input("Family History",min_value=0.0000,max_value=1.0000)
        AT = st.number_input("ATB7B Gene Mutation",min_value=0.0000,max_value=1.0000)
        Features = [[Age,Cl,cbs,Ciu,ALT,AST,TB,GGT,KFR,NSS,CFS,FH,AT]]
        if st.button("Predict"):
            y_pred = model.predict(Features)
            if y_pred == 1:
                st.warning("Detected high Probability of Wilsons Disease")
                st.title("Treatment and Management")
                st.subheader("How is Wilson disease treated?")
                st.markdown("""Treatment for Wilson disease focuses on lowering toxic levels of copper in your body and preventing organ damage and the symptoms you get when your organs aren’t functioning normally. 
                            Treatment includes: """)
                st.markdown("""1) Taking medicines that remove copper from the body (chelating agents, D-penicillamine, tetrathiomolybdate).""")
                st.markdown("""2) Taking zinc to prevent your intestines from absorbing copper """)
                st.markdown("""3) Eating a diet low in copper. """)
                st.title("Prevention")
                st.subheader("How can I prevent Wilson disease ?")
                st.markdown("""You can’t prevent Wilson disease since it’s the result of an inherited genetic mutation. If you have a family history of Wilson disease, talk with your healthcare provider about genetic testing to understand your risk of developing Wilson disease or having a child with this genetic condition. """)
            elif y_pred == 0:
                st.write("No Disease detected")
    if __name__ == '__main__':
        main()


