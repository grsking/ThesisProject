import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mimetypes
from PIL import Image
from scipy.signal import find_peaks

# Initialize the button state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Define a function to handle button click
def on_button_click():
    st.session_state.button_clicked = True

# Load the inference model
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model


# Make predictions
def inference(features, model):
    # if (model.predict(features)==0):
    #     return "This embryo has high chances of NOT being a BLASTO!"
    # else:
    #     return "This embryo has high chances of being a BLASTO!"
    return model.predict(features)

# Compute percentage of BLASTO and NOT BLASTO predictions
def compute_percentages(predictions):
    total = len(predictions)
    blasto_count = sum(predictions)
    not_blasto_count = total - blasto_count
    blasto_percentage = (blasto_count / total) * 100
    not_blasto_percentage = (not_blasto_count / total) * 100
    return blasto_percentage, not_blasto_percentage


# ---- Streamlit Configuration ----
st.set_page_config(page_title="AI-Based IVF Medical Device", layout="wide")
# Sidebar
st.title('**AI-Based IVF Medical Device**\n**Enhancing Embryo Selection with AI**')
st.markdown("**This is a Medical Device that integrates AI to enhance embryo selection accuracy using Machine Learning techniques.**")
image = Image.open('figures/blastocista.jpg')
st.image(image, use_container_width=True)


# Main
# ---- Step 1: Upload CSV ----
st.sidebar.header("1️⃣ Data Upload")
uploaded_file = st.sidebar.file_uploader("📂 Upload the CSV file with patient data", type=["csv"])

df = None  # Initialize df to None
df_embryo = None  # Initialize df_embryo to None
selected_patient = None  # Initialize unique_patients to None
model = None  # Initialize model to None
df_blasto = None  # Initialize df_blasto to None
df_not_blasto = None  # Initialize df_not_blasto to None

# ---- Step 3: Model loading ----
model_path = "models/model_rf_rf.joblib"
model = load_model(model_path)

if uploaded_file is not None:
    with st.spinner("Uploading data..."):
        try:
            uploaded_file.seek(0)  # Reset file pointer

            # Ensure the file is actually a CSV
            if not uploaded_file.name.endswith(".csv"):
                st.error("⚠ Uploaded file is not a CSV! Please check and re-upload.")
                st.stop()

            # Read CSV with debugging
            df = pd.read_csv(uploaded_file, encoding="utf-8")

            # Check if empty
            if df.empty:
                st.error("⚠ The uploaded CSV file is empty! Please upload a valid dataset.")
                st.stop()

            st.sidebar.success("✅ Data successfully uploaded!")

        except Exception as e:
            st.error(f"⚠ Error reading the CSV file: {e}")
            st.stop()  # Stop execution if the file is not valid

# ---- Step 2: Data Visualization (Only if Upload Successful) ----
if df is not None:
    st.header("2️⃣ Time Lapse of the Sum Mean Magnitude")

    if "patient_id" in df.columns:
        unique_patients = df["patient_id"].unique()
        selected_patient = st.sidebar.selectbox("🔬 Select a patient", unique_patients)
        df_patient = df[df["patient_id"] == selected_patient]
        df_embryo = df_patient.drop(["patient_id", "dish_well"], axis=1)

        # print all the rows of df_embryo
        st.write(f"📊 Patient: {selected_patient}, Number of embryos: {len(df_embryo)}:")
        st.write(df_embryo.head(len(df_embryo)))

        if not df_embryo.empty:
            st.write("📊 Displaying the time lapse for the selected patient:")
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df_embryo.T, color="green")
            # set x-axis labels every 24 hours
            plt.xticks(ticks=range(0, len(df_embryo.columns), 24), labels=range(0, len(df_embryo.columns), 24))
            plt.xlabel("Time (1/4 hours)")
            plt.ylabel("Value")
            plt.title(f"Sum Mean Magnitude Time Lapse for Patient {selected_patient}")
            st.pyplot(plt)
        else:
            st.error("⚠ No data found for the selected patient and embryo combination.")

        # ---- Step 4: AI-Based Prediction ----
        st.header("3️⃣ AI-Based Embryo Status Prediction")

        # Create the button and assign the click handler
        if st.button("🔍 Predict Embryo Status", on_click=on_button_click):
            pass
        # Check the button state and perform actions accordingly
        if st.session_state.button_clicked:
        #if st.button("🔍 Predict Embryo Status"):
            with st.spinner("Analyzing Time Series..."):
                if model is None:
                    st.error("⚠ Please upload and select a model first.")
                elif df_embryo is None:
                    st.error("⚠ Please upload the CSV file and select a patient first.")
                else:
                    predictions = inference(df_embryo.to_numpy(), model)
                    blasto_percentage, not_blasto_percentage = compute_percentages(predictions)
                    # Create a new dataframe with the dish_well column and the predictions
                    df_embryo_prediction = pd.DataFrame({"dish_well": df_patient["dish_well"], "prediction": predictions})
                    # replace the values of the column prediction with the string "BLASTO" or "NOT BLASTO"
                    df_embryo_prediction["prediction"] = df_embryo_prediction["prediction"].replace({1: "BLASTO", 0: "NOT BLASTO"})
                    # get from column dish_well the string representing the dish well number in the format "Dish Well X"
                    dish_well = df_embryo_prediction["dish_well"].iloc[0]
                    # remove form dish_well the last 2 characters
                    dish_well = dish_well[:-2]
                    # remove from column dish_well all the characters except the last one
                    df_embryo_prediction["dish_well"] = df_embryo_prediction["dish_well"].str[-1]
                    st.sidebar.write(f"Dish Well Label {dish_well}")
                    st.sidebar.write(f"📊 Predictions:")
                    # write the dataframe as a table
                    st.sidebar.write(df_embryo_prediction)

                    # Display the predictions as a pie chart with a size 50% of the page
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie([blasto_percentage, not_blasto_percentage], labels=["BLASTO", "NOT BLASTO"],
                           autopct="%1.1f%%", startangle=90)
                    ax.axis("equal")
                    st.sidebar.pyplot(fig)

                    # Display the percentage of BLASTO and NOT BLASTO predictions
                    st.sidebar.write(f"Percentage of BLASTO: {blasto_percentage:.2f}%")
                    st.sidebar.write(f"Percentage of NOT BLASTO: {not_blasto_percentage:.2f}%")

                    # create 2 dataframes with the predictions of the embryos each for each class
                    df_blasto = df_embryo[df_embryo_prediction["prediction"] == "BLASTO"]
                    df_not_blasto = df_embryo[df_embryo_prediction["prediction"] == "NOT BLASTO"]

                    # compute the avegage values of the embryos column by column for each class
                    avg_blasto = df_blasto.mean(axis=0)
                    avg_not_blasto = df_not_blasto.mean(axis=0)

                    # compute the stabdard deviation of the values of the embryos column by column for each class
                    std_blasto = df_blasto.std(axis=0)
                    std_not_blasto = df_not_blasto.std(axis=0)

            # ---- Step 5: Data Visualization ----
            st.header("4️⃣ Data Visualization")

            peak_height = st.slider("Peak height", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
            #peak_threshold = st.slider("Peak threshold", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
            peak_distance = st.slider("Peak distance", min_value=1, max_value=50, value=1, step=1)

            # find all relevant peaks in the average values of the embryos for each class
            peaks_blasto, _ = find_peaks(avg_blasto, height=peak_height, distance=peak_distance)
            peaks_not_blasto, _ = find_peaks(avg_not_blasto, height=peak_height, distance=peak_distance)
            # plot avg_blasto and avg_not_blasto and std_blasto and std_not_blasto
            plt.figure(figsize=(12, 6))
            plt.plot(avg_blasto, label="Average Time Lapse BLASTO", color="green")
            plt.fill_between(avg_blasto.index, avg_blasto - std_blasto, avg_blasto + std_blasto, alpha=0.2,
                             color="green")
            plt.plot(avg_not_blasto, label="Average Time Lapse NOT BLASTO", color="red")
            plt.fill_between(avg_not_blasto.index, avg_not_blasto - std_not_blasto, avg_not_blasto + std_not_blasto,
                             alpha=0.2, color="red")
            # plot the peaks
            plt.plot(peaks_blasto, avg_blasto.iloc[peaks_blasto], "o", color="green")
            plt.plot(peaks_not_blasto, avg_not_blasto.iloc[peaks_not_blasto], "o", color="red")
            plt.xticks(ticks=range(0, len(df_embryo.columns), 24), labels=range(0, len(df_embryo.columns), 24))
            plt.legend(loc="upper left")
            plt.xlabel("Time (1/4 hours)")
            plt.ylabel("Value")
            plt.title(f"Average Sum Mean Magnitude Time Lapse for Patient {selected_patient}")
            st.pyplot(plt)
