

# 🚀 AI-Based Disease Prediction System 🏥
🔬 *An AI-powered system to predict Diabetes, Kidney Disease, and COVID-19 based on patient health metrics.*  

---

## **📌 Features**  
✅ **Predicts Diabetes, Kidney Disease, and COVID-19** using patient symptoms  
✅ **Train AI Models on Custom Datasets** (Predict other diseases by adding new data)  
✅ **Machine Learning Models:** Logistic Regression & Random Forest  
✅ **Trained on Real-World Medical Datasets**  
✅ **User-Friendly CLI-Based Predictions** *(Can be expanded into a Web App)*  

---

## **🛠️ Tech Stack**  
🔹 **Python** 🐍  
🔹 **Pandas, NumPy** (Data Handling)  
🔹 **Scikit-learn** (ML Models)  
🔹 **Joblib** (Model Storage)  
🔹 **Flask/Streamlit** *(Optional for Web App Deployment)*  

---

## **📊 Model Performance**  
| **Disease**        | **Model Used**        | **Accuracy**  |  
|--------------------|----------------------|--------------|  
| 🩸 **Diabetes**    | Logistic Regression  | **85%**      |  
| 🏥 **Kidney Disease** | Random Forest     | **89%**      |  
| 😷 **COVID-19**    | Logistic Regression  | **90%**      |  

---

## **📂 Installation & Setup**  

### **1️⃣ Install Dependencies**  
```bash
pip install pandas numpy scikit-learn joblib
```

### **2️⃣ Run the Program**  
```bash
python main_file.py
```

### **3️⃣ Choose a Disease to Predict:**  
```
1 - Diabetes  
2 - COVID-19  
3 - Kidney Disease  
```

### **4️⃣ Enter Your Health Details & Get Predictions!**  
The system will analyze your inputs and provide an **AI-powered diagnosis**.  

---

## **🧑‍⚕️ Custom Disease Prediction (Train AI on New Diseases)**  
This system also allows you to **train a machine learning model on your own dataset**. If you have data for a new disease, you can **train a custom AI model** using the `train_and_predict()` function.  

### **How to Train a Model for a New Disease?**  
1️⃣ **Prepare Your Dataset**  
- Ensure your dataset is in CSV format and contains:  
  - **Feature columns** (patient symptoms, test results, etc.)  
  - **A target column** (0 = No Disease, 1 = Disease Present)  

2️⃣ **Run the Training Function**  
```python
from training import train_and_predict
train_and_predict("your_disease_dataset.csv")
```

3️⃣ **Follow the On-Screen Instructions**  
- The program will ask for the **target column** and **feature columns**.  
- It will train a **Random Forest model** on your data and save it as `trained_model.pkl`.  
- You can then **predict new cases** using the trained model.  

🔹 This feature makes the system **scalable** for other diseases like **heart disease, liver disorders, or rare medical conditions**!  

---

## **🚀 Future Improvements**  
🔹 **Deploy as a Flask/Streamlit Web App** 🌐  
🔹 **Enhance Accuracy with Deep Learning (LSTM, CNNs)** 🤖  
🔹 **Improve Dataset with Real-Time Patient Data Collection** 📊  

---

## **📩 Contact**  
💡 **Created by:** Aditya Sarohaa  
📧 **Email:** [adityasarohaa55@gmail.com](mailto:adityasarohaa55@gmail.com)  
🔗 **LinkedIn:** [linkedin.com/in/aditya-sarohaa-345336323](https://linkedin.com/in/aditya-sarohaa-345336323)  

---

## **📜 License & Usage**  
📜 **Copyright © 2024 Aditya Sarohaa**  
🔹 This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) License** – see the LICENSE file for details.  
🔹 **You may view this code, but you may NOT use, modify, distribute, or profit from it without explicit permission from the author.**  
