#######################################################
import uuid ## random id generator
from streamlit_option_menu import option_menu
from settings import *
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#######################################################
## Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
################################################### Defining Static Data ###############################################

image_link      = "https://wp.inews.co.uk/wp-content/uploads/2022/03/SEI_95866611.jpg?resize=640,360&strip=all&quality=90"
user_color      = '#000000'
title_webapp    = "Visitor Monitoring Web-Application"
html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            <img src = "{image_link}" align="right" width=150px ></h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

###################### Defining Static Paths ###################4
if st.sidebar.button('Click to Clear out all the data'):
    ## Clearing Visitor Database
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)
    ## Clearing Visitor History
    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
# st.write(VISITOR_HISTORY)


class FaceRecognitionProcessor(VideoTransformerBase):
    def __init__(self):
        # Initialize any variables or resources needed
        self.visitor_id = uuid.uuid1()
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        bytes_data = cv2.imencode('.jpg', frame)[1].tobytes()

        # convert image from OpenCV format to np.array
        image_array = cv2.imdecode(np.frombuffer(
            bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_array_copy = image_array.copy()

        # Perform face detection and recognition on the frame
        # Replace this with your face detection and recognition logic
        ## Validating Image
        # Detect faces in the loaded image
        max_faces = 0
        rois = []  # region of interests (arrays of face areas)

        ## To get location of Face from Image
        face_locations = face_recognition.face_locations(image_array)
        ## To encode Image to numeric format
        encodesCurFrame = face_recognition.face_encodings(
            image_array, face_locations)

        ## Generating Rectangle Red box over the Image
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # Save face's Region of Interest
            rois.append(image_array[top:bottom, left:right].copy())

            # Draw a box around the face and label it
            cv2.rectangle(image_array, (left, top), (right, bottom),
                          (0, 0, 255), 2)
            cv2.rectangle(image_array, (left, bottom + 35), (right, bottom),
                          (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25),
                        font, .55, (255, 255, 255), 1)

        ## Number of Faces identified
        max_faces = len(face_locations)

        if max_faces > 0:
            col1, col2 = st.columns(2)

            # select selected faces in the picture
            face_idxs = col1.multiselect("Select face#", range(max_faces),
                                         default=range(max_faces))

            ## Filtering for similarity beyond threshold
            similarity_threshold = col2.slider('Select Threshold for Similarity',
                                               min_value=0.0, max_value=1.0,
                                               value=0.5)

            flag_show = False

            if (col1.checkbox('Click to proceed!')) & (len(face_idxs) > 0):
                dataframe_new = pd.DataFrame()

                ## Iterating faces one by one
                for face_idx in face_idxs:
                    ## Getting Region of Interest for that Face
                    roi = rois[face_idx]

                    # initial database for known faces
                    database_data = initialize_data()

                    ## Getting Available information from Database
                    face_encodings = database_data[COLS_ENCODE].values
                    dataframe = database_data[COLS_INFO]

                    # Comparing ROI to the faces available in database and finding distances and similarities
                    faces = face_recognition.face_encodings(roi)

                    if len(faces) < 1:
                        ## Face could not be processed
                        st.error(f'Please Try Again for face#{face_idx}!')
                    else:
                        face_to_compare = faces[0]
                        ## Comparing Face with available information from database
                        dataframe['distance'] = face_recognition.face_distance(
                            face_encodings, face_to_compare)
                        dataframe['distance'] = dataframe['distance'].astype(
                            float)

                        dataframe['similarity'] = dataframe.distance.apply(
                            lambda distance: f"{face_distance_to_conf(distance):0.2}")
                        dataframe['similarity'] = dataframe['similarity'].astype(
                            float)

                        dataframe_new = dataframe.drop_duplicates(
                            keep='first')
                        dataframe_new.reset_index(drop=True, inplace=True)
                        dataframe_new.sort_values(
                            by="similarity", ascending=True)

                        dataframe_new = dataframe_new[dataframe_new['similarity']
                                                      > similarity_threshold].head(1)
                        dataframe_new.reset_index(drop=True, inplace=True)

                        if dataframe_new.shape[0] > 0:
                            (top, right, bottom, left) = (
                                face_locations[face_idx])

                            ## Save Face Region of Interest information to the list
                            rois.append(
                                image_array_copy[top:bottom, left:right].copy())

                            # Draw a Rectangle Red box around the face and label it
                            cv2.rectangle(image_array_copy, (left, top),
                                          (right, bottom), (0, 0, 255), 2)
                            cv2.rectangle(image_array_copy, (left, bottom + 35),
                                          (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25),
                                        font, .55, (255, 255, 255), 1)

                            ## Getting Name of Visitor
                            name_visitor = dataframe_new.loc[0, 'Name']
                            attendance(self.visitor_id, name_visitor)

                            flag_show = True

                        else:
                            st.error(
                                f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                            st.info(
                                'Please Update the database for a new person or click again!')
                            attendance(self.visitor_id, 'Unknown')

                if flag_show == True:
                    st.image(BGR_to_RGB(image_array_copy), width=720)

        else:
            st.error('No human face detected.')

        return image_array
      
 

def main():
    ###################################################
    st.sidebar.header("About")
    st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
                    "Webapp using 'Face Recognition' and Streamlit")
    ###################################################

    selected_menu = option_menu(None,
        ['Visitor Validation', 'Live Visitor Validation', 'View Visitor History', 'Add to Database','View Confusion Matrix'],
        icons=['camera', "clock-history", 'person-plus', 'bar-chart'],
        ## icons from website: https://icons.getbootstrap.com/
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Visitor Validation':
        actual_name = st.text_input("Enter the actual name of the visitor (for testing purposes):")
        ## Generates a Random ID for image storage
        visitor_id = uuid.uuid1()

        ## Reading Camera Image
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()

            # convert image from opened file to np.array
            image_array         = cv2.imdecode(np.frombuffer(bytes_data,
                                                             np.uint8),
                                               cv2.IMREAD_COLOR)
            image_array_copy    = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # st.image(cv2_img)

            ## Saving Visitor History
            with open(os.path.join(VISITOR_HISTORY,
                                   f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                ## Validating Image
                # Detect faces in the loaded image
                max_faces   = 0
                rois        = []  # region of interests (arrays of face areas)

                ## To get location of Face from Image
                face_locations  = face_recognition.face_locations(image_array)
                ## To encode Image to numeric format
                encodesCurFrame = face_recognition.face_encodings(image_array,
                                                                  face_locations)

                ## Generating Rectangle Red box over the Image
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    # Save face's Region of Interest
                    rois.append(image_array[top:bottom, left:right].copy())

                    # Draw a box around the face and label it
                    cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                ## Showing Image
                st.image(BGR_to_RGB(image_array), width=720)

                ## Number of Faces identified
                max_faces = len(face_locations)

                if max_faces > 0:
                    col1, col2 = st.columns(2)

                    # select selected faces in the picture
                    face_idxs = col1.multiselect("Select face#", range(max_faces),
                                                 default=range(max_faces))

                    ## Filtering for similarity beyond threshold
                    similarity_threshold = col2.slider('Select Threshold for Similarity',
                                                         min_value=0.0, max_value=1.0,
                                                         value=0.5)
                                                    ## check for similarity confidence greater than this threshold

                    flag_show = False

                    if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
                        dataframe_new = pd.DataFrame()

                        ## Iterating faces one by one
                        for face_idx in face_idxs:
                            ## Getting Region of Interest for that Face
                            roi = rois[face_idx]
                            # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

                            # initial database for known faces
                            database_data = initialize_data()
                            # st.write(DB)

                            ## Getting Available information from Database
                            face_encodings  = database_data[COLS_ENCODE].values
                            dataframe       = database_data[COLS_INFO]

                            # Comparing ROI to the faces available in database and finding distances and similarities
                            faces = face_recognition.face_encodings(roi)
                            # st.write(faces)

                            if len(faces) < 1:
                                ## Face could not be processed
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = faces[0]
                                ## Comparing Face with available information from database
                                dataframe['distance'] = face_recognition.face_distance(face_encodings,
                                                                                       face_to_compare)
                                dataframe['distance'] = dataframe['distance'].astype(float)

                                dataframe['similarity'] = dataframe.distance.apply(
                                    lambda distance: f"{face_distance_to_conf(distance):0.2}")
                                dataframe['similarity'] = dataframe['similarity'].astype(float)

                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True)

                                dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)

                                if dataframe_new.shape[0]>0:
                                    (top, right, bottom, left) = (face_locations[face_idx])

                                    ## Save Face Region of Interest information to the list
                                    rois.append(image_array_copy[top:bottom, left:right].copy())

                                    # Draw a Rectangle Red box around the face and label it
                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                    ## Getting Name of Visitor
                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(visitor_id, name_visitor)

                                    flag_show = True

                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(visitor_id, 'Unknown')

                        if flag_show == True:
                            st.image(BGR_to_RGB(image_array_copy), width=720)

                else:
                    st.error('No human face detected.')
                    
    if selected_menu == 'Live Visitor Validation':
        st.write("Live Visitor Validation")
        webrtc_streamer(key="example",
                         video_processor_factory=FaceRecognitionProcessor,
                         async_processing=True)
        

    
    if selected_menu == 'View Visitor History':
        st.write("View Visitor History")
        view_attendace()
        if st.button("Generate Confusion Matrix"):
            # Example for processing previously saved data
            true_labels = []  # Populate these with your data
            predicted_labels = []  # Populate these with your data
            st.write("Button clicked!")
            generate_confusion_matrix(true_labels, predicted_labels)

    

    if selected_menu == 'Add to Database':
        col1, col2, col3 = st.columns(3)

        face_name  = col1.text_input('Name:', '')
        pic_option = col2.radio('Upload Picture',
                                options=["Upload a Picture",
                                         "Click a picture"])

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture',
                                                 type=allowed_image_type)
            if img_file_buffer is not None:
                # To read image file buffer with OpenCV:
                file_bytes = np.asarray(bytearray(img_file_buffer.read()),
                                        dtype=np.uint8)

        elif pic_option == 'Click a picture':
            img_file_buffer = col3.camera_input("Click a picture")
            if img_file_buffer is not None:
                # To read image file buffer with OpenCV:
                file_bytes = np.frombuffer(img_file_buffer.getvalue(),
                                           np.uint8)

        if ((img_file_buffer is not None) & (len(face_name) > 1) &
                st.button('Click to Save!')):
            # convert image from opened file to np.array
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # st.write(image_array)
            # st.image(cv2_img)

            with open(os.path.join(VISITOR_DB,
                                   f'{face_name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                # st.success('Image Saved Successfully!')

            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array,
                                                              face_locations)

            df_new = pd.DataFrame(data=encodesCurFrame,
                                  columns=COLS_ENCODE)
            df_new[COLS_INFO] = face_name
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

            # st.write(df_new)
            # initial database for known faces
            DB = initialize_data()
            add_data_db(df_new)
    

#######################################################
if __name__ == "__main__":
    main()
#######################################################
