import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
# Initialize the YOLO model
model = YOLO('best.pt')
model.conf = 0.30
# Create the box annotator
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)
live = st.button("Start Live Detection")
feed = st.button("Upload File")
if feed:
# Streamlit file uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set up the VideoWriter object to save the output video
        output_file = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        # Streamlit container to display the video frames
        st_frame = st.empty()

        # Read and process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run the YOLO model on the frame
            result = model(frame)[0]
            
            # Convert YOLO results to supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Annotate the frame with detections
            frame = box_annotator.annotate(scene=frame, detections=detections)
            
            # Write the annotated frame to the output video
            if frame is not None:
                out.write(frame)
            
            # Convert the frame to RGB format for displaying with Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in Streamlit
            st_frame.image(frame_rgb, caption='feed')

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Notify the user and provide a link to download the video
        if output_file is not None:
            st.video(output_file)
        st.markdown(f"[Download the output video](./{output_file})")


elif live:
        cap = cv2.VideoCapture(0)
    
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set up the VideoWriter object to save the output video
        output_file = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        # Streamlit container to display the video frames
        st_frame = st.empty()

        # Read and process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run the YOLO model on the frame
            result = model(frame)[0]
            
            # Convert YOLO results to supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Annotate the frame with detections
            frame = box_annotator.annotate(scene=frame, detections=detections)
            
            # Write the annotated frame to the output video
            if frame is not None:
                out.write(frame)
            
            # Convert the frame to RGB format for displaying with Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in Streamlit
            st_frame.image(frame_rgb, caption='feed')

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Notify the user and provide a link to download the video
        st.success(f"Video saved as {output_file}")
        st.video(output_file)
        st.markdown(f"[Download the output video](./{output_file})")
