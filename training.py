import pandas as pd #for data handling
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
 # Load the dataset
df1 = pd.read_csv('diabetes.csv')
df2 = pd.read_csv('kidney_disease.csv')
df3 = pd.read_csv('covid-19_symptoms.csv')
# Function to get user input
def ask_value(columns):
    user_input = {}
    for column in columns:
        value = int(input(f'{column} = \t'))
        user_input[column] = value
    return user_input

def diabetes():
    # Features and target
    X = df1[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']]
    y = df1['Outcome']

    # Convert y to 1D
    y = y.values.ravel()

    # Split the data
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Testing the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy score:', accuracy)
    
    # Save the model
    joblib.dump(model, 'diabetes_model.pkl')

def execute_diabetes():
    # User input for prediction
    print("\nEnter the following details for diabetes prediction:")
    Pregnancies = int(input('Enter number of pregnancies: '))
    Glucose = int(input('Enter your glucose level: '))
    BloodPressure = int(input('Enter your blood pressure: '))
    DPF_ask = int(input('Do you possess your specific dpf value: 1-Yes, 2-No '))
    
    if DPF_ask == 1:
        DiabetesPedigreeFunction = float(input('Enter your Diabetes Pedigree function: '))
    else:
        DiabetesPedigreeFunction = df1['DiabetesPedigreeFunction'].median()
        print(f"Using default Diabetes Pedigree Function value: {DiabetesPedigreeFunction}")

    SkinThickness = int(input('Enter your skin thickness: '))
    Insulin = int(input('Enter your insulin level: '))
    BMI = float(input('Enter your BMI: '))
    Age = int(input('Enter your age: '))

    # Prepare the new data as a DataFrame
    X_new = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]], 
                         columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])
    
    # Load the model and predict
    model = joblib.load('diabetes_model.pkl')
    y_pred_new = model.predict(X_new)
    #printing a user friendly output
    if y_pred_new[0] == 1:
        print("You are very likely to be diabetic. Visit a doctor.")
    else:
        print('Stay happy, you are not diabetic.')


# Kidney disease prediction
def kidney_disease():
    global W

    # Encoding categorical values
    encoder = LabelEncoder()
    df2['Hypertension_rating'] = encoder.fit_transform(df2['Hypertension'])
    df2['Diabetes_Mellitus_rating'] = encoder.fit_transform(df2['Diabetes_Mellitus'])
    df2['Coronary_Artery_Disease_rating'] = encoder.fit_transform(df2['Coronary_Artery_Disease'])
    df2['Appetite_rating'] = encoder.fit_transform(df2['Appetite'])
    df2['Pedal_Edema_rating'] = encoder.fit_transform(df2['Pedal_Edema'])
    df2['final_outcome'] = encoder.fit_transform(df2['Disease_Classification'])

    # Features and target
    W = df2[['Hypertension_rating', 'Diabetes_Mellitus_rating', 'Coronary_Artery_Disease_rating', 'Appetite_rating', 'Pedal_Edema_rating']]
    z = df2['final_outcome']

    # Train-test split
    W_train, W_test, z_train, z_test = tts(W, z, test_size=0.2, random_state=42)

    # Train the model
    model.fit(W_train, z_train)

    # Evaluate the model
    z_pred = model.predict(W_test)
    accuracy_score_value = accuracy_score(z_test, z_pred)
    print('Accuracy of the model:', accuracy_score_value)

    joblib.dump(model, 'kidney_disease_model.pkl')

   
def execute_kidney_disease():
    # Get user input and make a prediction
    user_input = ask_value(W.columns)
    W_new = pd.DataFrame([user_input], columns=W.columns)

    # Align the new data's features with the trained model's feature names
    W_new = W_new.reindex(columns=model.feature_names_in_, fill_value=0)

    z_pred_new = model.predict(W_new)
    if z_pred_new[0]==1:
        print('You are having chronic kidney disease.')
    elif z_pred_new[0]==0:
        print('You are not having chronic kidney disease.')
# COVID-19 symptom checker prediction
def covid_predict():
    global C
    C = df3[df3.columns[:-1]]
    # Add Covid Positive column based on conditions
    df3['Covid_positive'] = 0
    df3.loc[(df3['Severity_Severe']==1)|((df3['Severity_Moderate']==1)&((df3['Fever']==1)|(df3['Dry-Cough']==1)|(df3['Difficulty-in-Breathing']==1)))|(df3['Contact_Yes']==1),'Covid_positive']=1

    # Features and target
    
    d = df3['Covid_positive']

    # Train-test split
    C_train, C_test, d_train, d_test = tts(C, d, test_size=0.2, random_state=42)

    # Train the model
    model.fit(C_train, d_train)

    # Evaluate the model
    d_pred = model.predict(C_test)
    print('Accuracy score for covid dataset:', accuracy_score(d_test, d_pred))

    joblib.dump(model, 'covid_model.pkl')
def execute_covid_predict():
    # Get user input and make a prediction
    print('Anser only in 0 and 1 . 0=False,1=True for accurate predictions ')
    user_input_covid = ask_value(C.columns)
    C_new = pd.DataFrame([user_input_covid], columns=C.columns)
    model = joblib.load('covid_model.pkl')#loading the model for predictions
    d_pred_new = model.predict(C_new)
    if d_pred_new[0] == 1:
        print('You are corona positive.')
    else:
        print('You are corona negative.')

# Run all models
kidney_disease()
covid_predict()
diabetes()

def train_and_predict(data_file):
    # Load the dataset
    df = pd.read_csv(data_file)
    print("Columns in the dataset:", list(df.columns))
    
    # Ask the user for the target and features
    target_column = input("Enter the target column name: ").strip()
    feature_columns = input("Enter the feature column names (comma-separated): ").strip().split(",")
    feature_columns = [col.strip() for col in feature_columns]
    
    # Ensure the target and features exist in the dataset
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in the dataset.")
    
    # Extract features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
    
    # Train a model (Random Forest in this case)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the model
    model_filename = "trained_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")
    
    # Function to take user input for prediction
    def get_user_input(columns):
        user_input = []
        print("\nEnter values for the following features:")
        for col in columns:
            value = float(input(f"{col}: ").strip())
            user_input.append(value)
        return user_input
    
    # Predict on new data
    user_input_data = get_user_input(feature_columns)
    user_input_df = pd.DataFrame([user_input_data], columns=feature_columns)
    prediction = model.predict(user_input_df)
    print("\nPrediction for the input:", prediction[0])
    
data = {
    "Age": [25, 45, 35, 50, 40, 60, 30, 55],
    "BMI": [22.5, 27.8, 24.0, 30.5, 26.2, 32.1, 23.0, 29.4],
    "BloodPressure": [120, 140, 130, 150, 135, 160, 125, 145],
    "Glucose": [85, 105, 95, 110, 100, 115, 90, 108],
    "Disease": [0, 1, 0, 1, 0, 1, 0, 1],
}

df = pd.DataFrame(data)
df.to_csv("example_data.csv", index=False)
print("Example dataset saved as 'example_data.csv'")
train_and_predict('example_data.csv')