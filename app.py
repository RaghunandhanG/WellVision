import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import tempfile

# Set the maximum file size (in bytes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# Initialize the YOLO model
model = YOLO('best.pt')

# Streamlit slider to set the confidence threshold
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create the box annotator
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Check the file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the maximum limit of 500 MB. Your file size is {uploaded_file.size / (1024 * 1024):.2f} MB.")
    else:
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
            
            # Run the YOLO model on the frame with the confidence threshold
            results = model(frame, conf=confidence_threshold)[0]
            
            # Convert YOLO results to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Annotate the frame with detections
            frame = box_annotator.annotate(scene=frame, detections=detections)
            
            # Write the annotated frame to the output video
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
