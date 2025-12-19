from pickle import load
import streamlit as st


model = load(open('data/models/12_nlp_url_spam.pkl', 'rb'))

class_dict = {"0": "url",
              "1": "is_spam"}

st.title("Spam - Model prediction")
st.markdown("""Power by: [Saray Ruiz](https://sarayruiz.net)""")
st.divider()

val1 = st.text_input("Introduzca URL", value="", max_chars=None, key=None, type="default", help="Introduce desde htpp...",
                    label_visibility="visible")
    
    

if st.button("Predict"):
    prediction = str(model.predict([val1])[0])
    pred_class = class_dict[prediction]
    st.divider()
    st.write("Prediction:", pred_class)
    st.divider()
