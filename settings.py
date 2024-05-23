import os, pathlib
import streamlit as st
import os, datetime, json, sys, pathlib, shutil
import pandas as pd
import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
########################################################################################################################
# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()

OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
# st.write(VISITOR_DB)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
# st.write(VISITOR_HISTORY)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
########################################################################################################################
## Defining Parameters

COLOR_DARK  = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

## Database
data_path       = VISITOR_DB
file_db         = 'visitors_db.csv'         ## To store user information
file_history    = 'visitors_history.csv'    ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
################################################### Defining Function ##############################################
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

#################################################################
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st

def generate_confusion_matrix(y_true, y_pred):
    # Load the Excel file into a DataFrame
    visitor_database = pd.read_excel("myenv/visitor_database/visitors_db.csv", engine='openpyxl')
    y_true = visitor_database["Name"].tolist()

    st.write("Generating confusion matrix...")
    st.write("True labels:", y_true)
    st.write("Predicted labels:", y_pred)

    if not y_true or not y_pred:
        st.error("True labels or predicted labels are empty.")
        return

    if len(y_true) != len(y_pred):
        st.error("True labels and predicted labels lengths do not match.")
        return

    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert np.isfinite(y_true).all(), "True labels contain non-finite values"
        assert np.isfinite(y_pred).all(), "Predicted labels contain non-finite values"

        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating confusion matrix: {e}")

def add_data_db(df_visitor_details):
    try:
        if os.path.exists(os.path.join(data_path, file_db)):
            df_all = pd.read_csv(os.path.join(data_path, file_db))
        else:
            df_all = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

        df_all = pd.concat([df_all, df_visitor_details], ignore_index=True)
        df_all.drop_duplicates(keep='first', inplace=True)
        df_all.to_csv(os.path.join(data_path, file_db), index=False)
        st.success('Details Added Successfully!')

    except Exception as e:
        st.error(e)

#################################################################
# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

#################################################################
def findEncodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

#################################################################
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power(
            (linear_val - 0.5) * 2, 0.2))

#################################################################
def attendance(id, name, actual_name=None):
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={"id": [id], "visitor_name": [name], "actual_name": [actual_name], "Timing": [dtString]})

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendance = pd.read_csv(f_p)
        df_attendance = pd.concat([df_attendance, df_attendace_temp], ignore_index=True)
        df_attendance.to_csv(f_p, index=False)

#################################################################
def view_attendace():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    # st.write(f_p)
    df_attendace_temp = pd.DataFrame(columns=["id",
                                              "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace_temp = pd.read_csv(f_p)

    df_attendace = df_attendace_temp.sort_values(by='Timing',
                                                 ascending=False)
    df_attendace.reset_index(inplace=True, drop=True)

    st.write(df_attendace)

    if df_attendace.shape[0]>0:
        id_chk  = df_attendace.loc[0, 'id']
        id_name = df_attendace.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID',
                                    options=['None']+list(df_attendace['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if ((file.endswith(tuple(allowed_image_type))) &
                                                                              (file.startswith(selected_img) == True))]

        if len(avail_files)>0:
            selected_img_path = os.path.join(VISITOR_HISTORY,
                                             avail_files[0])
            #st.write(selected_img_path)

            ## Displaying Image
            st.image(Image.open(selected_img_path))


########################################################################################################################