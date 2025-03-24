
import pickle
import streamlit as st

# Set the page title and description
st.title("UCLA Admission Chance Predictor")
st.write("""
This app predicts a student's chance of admission into UCLA
based on various personal characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
nn_pickle = open("models/NNmodel.pkl", "rb")
nn_model = pickle.load(nn_pickle)
nn_pickle.close()


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")
    
    # GRE Score input
    GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340)
    
    # TOFEL Score input
    TOFEL_Score = st.number_input("TOFEL Score", min_value=0, max_value=120)
    
    # University Rating
    University_Rating = st.selectbox("University_Rating", options=["1","2","3","4","5"])
    
    # SOP
    SOP = st.number_input("Purpose Strength", min_value=1, max_value=5)
    
    # LOR
    LOR = st.number_input("Letter of Recomendation Strength", min_value=1, max_value=5)
    
    # CGPA
    CGPA = st.number_input("Student's Undergraduate GPA", min_value=0, max_value=10)
    
    # Research
    Research = st.selectbox("Research Experience", options=["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")


# Handle the dummy variables to pass to the model
if submitted:

    University_Rating_1 = 1 if University_Rating == "1" else 0
    University_Rating_2 = 1 if University_Rating == "2" else 0
    University_Rating_3 = 1 if University_Rating == "3" else 0
    University_Rating_4 = 1 if University_Rating == "4" else 0
    University_Rating_5 = 1 if University_Rating == "5" else 0

    Research_0 = 1 if Research == "Yes" else 0
    Research_1 = 1 if Research == "No" else 0


    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = [[GRE_Score, TOFEL_Score, University_Rating_1, University_Rating_2,
                         University_Rating_3, University_Rating_4, University_Rating_5, 
                         SOP, LOR, CGPA, Research_0, Research_1
    ]]

    # Make prediction
    new_prediction = nn_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    if new_prediction[0] == 1:
        st.write("You have chance for UCLA !")
    else:
        st.write("Sorry, you don't have chance for UCLA.")

st.write(
    """We used a machine learning (Neural Network) model to predict your chance."""
)
st.image("loss_curve.png")
