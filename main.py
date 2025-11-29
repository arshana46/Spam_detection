import streamlit as st
import pickle
import base64

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
model = pickle.load(open("Multinomial model", "rb"))
vectorizer = pickle.load(open("Count Vectorizer.pkl", "rb"))

# -----------------------------------
# Optional Background Image Function
# -----------------------------------
def add_bg_image(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Uncomment if you want background
add_bg_image("spam_detection.jpg")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📧 Spam Detection App")
st.write("Enter a message to classify it as **Spam** or **Not Spam**.")

# Input Box
user_input = st.text_area("Enter the message here:")

# Predict Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Convert text to vector
        transformed = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")

# Footer
st.markdown("---")
st.caption("Spam Detection Model • Streamlit Deployment")
