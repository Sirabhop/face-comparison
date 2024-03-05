import streamlit as st
import face_recognition as frc

st.title("Face Comparison Experiment")
st.write('For riders facial verification')
cut_off = st.slider('Cut-Off Confidence', min_value=0.0, max_value=1.0, value=0.8)

base_face_id = st.text_input("Place Your Rider ID")
if base_face_id:
    uploaded_face = st.file_uploader("Upload Your Face", type=['jpg','png','jpeg'], accept_multiple_files=False)
    # base_face = frc.load_image_file(f'/Users/sirabhopsaengumyoun/Desktop/My Project/face_comparision/face_database/{str(base_face_id)}.jpg')
    base_face = frc.load_image_file(f'face_database/{str(base_face_id)}.jpg')

    encoded_base_face = frc.face_encodings(base_face)
    if uploaded_face:
        face = frc.load_image_file(uploaded_face)
        encoded_comparing_face = frc.face_encodings(face)
        distance = frc.face_distance(encoded_comparing_face, encoded_base_face[0])

        if distance.size > 0:

            accuracy = 1 - distance[0]

            if accuracy >= cut_off:
                st.write(f':green[Your verification has been APPROVED with {round(accuracy*100, 2)} confidence.]')
            else:
                st.write(f':red[Your verification has been REJECTED with {round(accuracy*100, 2)} confidence.]', )
                st.write(':red[Sorry to see your dissapointment, you may want to re-upload.]')

        else:
            st.write(':red[Please upload a clear face.]')
