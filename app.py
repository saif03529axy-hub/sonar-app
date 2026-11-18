import os
import pickle
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "sonar_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("SONAR: Rock vs Mine Prediction")
st.write("60 sensor values comma (,) se alag karke neeche daalo.")

text = st.text_area("Sensor values", height=150)

if st.button("Predict"):
    try:
        values = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
        if len(values) != 60:
            st.error(f"60 values chahiye, aapne {len(values)} di hain.")
        else:
            arr = np.array(values).reshape(1, -1)
            pred = model.predict(arr)[0]
            if pred == "M":
                st.error("Result: Mine (M)")
            else:
                st.success("Result: Rock (R)")
    except ValueError:
        st.error("Sirf numbers likho, comma se alag karke.")
