# Copyright © 2024 Aditya Sarohaa
# Licensed under CC BY-NC-ND 4.0 (No modifications or commercial use allowed)


import joblib
import pandas as pd
from training import*
def load_diabetes_model():
    # Load the model from the file
    model = joblib.load('diabetes_model.pkl')
    return model

def load_kidney_model():
    model = joblib.load('kidney_disease_model.pkl')
    return model

def load_covid_model():
    model = joblib.load('covid_model.pkl')
    return model

# Example usage in the menu
def menu():
    print('Enter what health status do you want to predict: 1-Diabetes, 2-Covid, 3-Kidney Disease, any other key to exit')

    while True:
        choice = int(input('Enter your choice: '))

        if choice == 1:
            model = load_diabetes_model()  # Load the diabetes model
            execute_diabetes()

        elif choice == 2:
            
            model = load_covid_model()  # Load the covid model
            execute_covid_predict()

        elif choice == 3:
            
            model = load_kidney_model()  
            execute_kidney_disease()
            

        else:
            break

menu()

