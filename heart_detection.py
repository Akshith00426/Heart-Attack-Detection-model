import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the model and scaler
try:
    model = joblib.load("heart_model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Function to predict heart attack risk and provide tailored suggestions
def predict():
    try:
        # Get user input and prepare it for the model
        original_user_data = [
            float(age_entry.get()),
            float(sex_entry.get()),
            float(cp_entry.get()),
            float(trtbps_entry.get()),
            float(chol_entry.get()),
            float(fbs_entry.get()),
            float(restecg_entry.get()),
            float(thalachh_entry.get()),
            float(exng_entry.get()),
            float(oldpeak_entry.get()),
            float(slp_entry.get()),
            float(caa_entry.get()),
            float(thall_entry.get())
        ]
        
        # Tailored suggestions based on unscaled inputs
        suggestions = "General Suggestions:\n- Maintain a balanced diet\n- Engage in regular physical activity\n- Get regular checkups\n"
        
        # Condition checks before scaling
        if original_user_data[3] > 140:  # High Resting BP
            suggestions += "\n- Consider reducing salt intake to lower blood pressure.\n- Monitor your blood pressure regularly.\n"
        if original_user_data[4] > 200:  # High Cholesterol
            suggestions += "\n- Follow a low-cholesterol diet with more fruits and vegetables.\n- Avoid saturated fats and trans fats.\n"
        if original_user_data[5] == 1:  # High Fasting Blood Sugar
            suggestions += "\n- Maintain a healthy weight and consider regular exercise.\n- Monitor blood sugar levels.\n"
        if original_user_data[8] == 1:  # Exercise Induced Angina
            suggestions += "\n- Avoid high-intensity exercise; consider moderate-intensity activities.\n- Discuss exercise plans with your doctor.\n"
        if original_user_data[9] > 2.0:  # High Old Peak (ST depression)
            suggestions += "\n- Manage stress levels through activities like yoga or meditation.\n"
        
        # Reshape data to match model input and scale it
        user_data = np.array(original_user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        
        # Make prediction and get prediction probability
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)

        # Show prediction probability as a graph
        show_probability_graph(prediction_proba[0])

        # Display the result
        if prediction[0] == 0:
            result = "High Risk of Heart Attack"
            suggestions = f"{result}\n\n{suggestions}"
        else:
            result = "Low Risk of Heart Attack"
            suggestions = f"{result}\n\n{suggestions}\nKeep up the good work!"

        messagebox.showinfo("Prediction Result", f"{suggestions}\n\nPrediction Probability: High Risk: {prediction_proba[0][0]:.2f}, Low Risk: {prediction_proba[0][1]:.2f}")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to display the prediction probability graph
def show_probability_graph(probabilities):
    # Create a bar chart of the probabilities
    labels = ["High Risk", "Low Risk"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probabilities, color=['red', 'green'])
    plt.title("Prediction Probability")
    plt.xlabel("Risk")
    plt.ylabel("Probability")
    plt.ylim(0, 1)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(plt.gcf(), master=app)  # 'app' is the Tkinter window
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

# Initialize the tkinter window
app = tk.Tk()
app.title("Heart Attack Prediction")
app.geometry("500x700")

# Create a canvas with both horizontal and vertical scrollbars
canvas = tk.Canvas(app)
scrollbar_vertical = ttk.Scrollbar(app, orient="vertical", command=canvas.yview)
scrollbar_horizontal = ttk.Scrollbar(app, orient="horizontal", command=canvas.xview)

# Create the scrollable frame
scrollable_frame = ttk.Frame(canvas)

# Configure the scrollable frame
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# Create window for the scrollable frame inside the canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar_vertical.set, xscrollcommand=scrollbar_horizontal.set)

# Pack the canvas and both scrollbars
canvas.pack(side="left", fill="both", expand=True)
scrollbar_vertical.pack(side="right", fill="y")
scrollbar_horizontal.pack(side="bottom", fill="x")

# Input labels and entries in the scrollable frame
labels = ["Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting BP (in mm Hg)", 
          "Cholesterol (mg/dL)", "Fasting Blood Sugar (1 if >120 mg/dL, else 0)", "Rest ECG (0-2)", 
          "Max Heart Rate", "Exercise Induced Angina (1=Yes, 0=No)", "Old Peak", "Slope (0-2)", 
          "CA (0-3)", "Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)"]

entries = []

for i, label in enumerate(labels):
    tk.Label(scrollable_frame, text=label).grid(row=i, column=0, pady=5, sticky="w")
    entry = tk.Entry(scrollable_frame)
    entry.grid(row=i, column=1, pady=5, padx=10, sticky="w")
    entries.append(entry)

# Unpack entries for easy access
(age_entry, sex_entry, cp_entry, trtbps_entry, chol_entry, fbs_entry, restecg_entry, 
 thalachh_entry, exng_entry, oldpeak_entry, slp_entry, caa_entry, thall_entry) = entries

# Create Predict button
predict_button = tk.Button(scrollable_frame, text="Predict", command=predict, bg="lightblue")
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=20)

# Run the tkinter main loop
app.mainloop()
