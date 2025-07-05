import cv2
import mediapipe as mp
import os
import numpy as np
from ultralytics import YOLO
import urllib.request

def download_yolo_model():
    """Download YOLO model if not exists"""
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLO model (first time only)...")
        try:
            model = YOLO(model_path)  # This will auto-download
            print("YOLO model downloaded successfully!")
            return model
        except Exception as e:
            print(f"Error downloading YOLO model: {e}")
            return None
    else:
        return YOLO(model_path)

def detect_faces(source):
    """
    Enhanced face detection using both MediaPipe and YOLO for maximum accuracy
    """
    # Initialize MediaPipe face detection with lower confidence for better recall
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # Use model 1 for better long-range detection
        min_detection_confidence=0.3  # Lower threshold for better detection
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize YOLO model
    print("Loading YOLO model...")
    yolo_model = download_yolo_model()
    if yolo_model is None:
        print("Warning: YOLO model not available, using only MediaPipe")
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
    
    # Get video properties
    if source != 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Info - FPS: {fps}, Resolution: {width}x{height}, Frames: {frame_count}")
        
        # Calculate display size (scale down if too large)
        display_width = width
        display_height = height
        max_display_width = 1280  # Maximum width for display
        max_display_height = 720  # Maximum height for display
        
        if width > max_display_width or height > max_display_height:
            # Calculate scale factor to fit screen
            scale_w = max_display_width / width
            scale_h = max_display_height / height
            scale = min(scale_w, scale_h)
            
            display_width = int(width * scale)
            display_height = int(height * scale)
            print(f"Display size adjusted to: {display_width}x{display_height} (scale: {scale:.2f})")
        
        # Store scale factor for resizing
        display_scale = min(1.0, min(max_display_width / width, max_display_height / height))
    else:
        display_scale = 1.0
    

    print(f"Processing video from: {'Camera' if source == 0 else source}")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_number = 0
    total_faces_detected = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            if source == 0:
                print("Error: Could not read from camera")
                break
            else:
                print(f"End of video file reached. Total faces detected: {total_faces_detected}")
                break
        
        frame_number += 1
        original_frame = frame.copy()
        
        # Method 1: MediaPipe Detection
        mp_faces = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = mp_face_detection.process(rgb_frame)
        
        if mp_results.detections:
            for detection in mp_results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                mp_faces.append(bbox)
                
                # Draw MediaPipe detection in green
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                cv2.putText(frame, f'MP: {detection.score[0]:.2f}', 
                           (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Method 2: YOLO Detection
        yolo_faces = []
        if yolo_model is not None:
            try:
                # Run YOLO detection
                results = yolo_model(frame, conf=0.25, classes=[0])  # class 0 is 'person'
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # Convert to format (x, y, w, h)
                            bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                            yolo_faces.append(bbox)
                            
                            # Draw YOLO detection in blue
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(frame, f'YOLO: {conf:.2f}', 
                                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"YOLO detection error: {e}")
        
        
        # Count total unique faces
        all_faces = mp_faces + yolo_faces
        total_current_faces = len(all_faces)
        
        if total_current_faces > 0:
            total_faces_detected += total_current_faces
            print(f"Frame {frame_number}: {total_current_faces} faces detected "
                  f"(MP: {len(mp_faces)}, YOLO: {len(yolo_faces)})")
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_number}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Faces: {total_current_faces}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        display_frame = frame.copy()
        
        # Resize frame for display if needed
        if source != 0 and display_scale < 1.0:
            new_width = int(frame.shape[1] * display_scale)
            new_height = int(frame.shape[0] * display_scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        
        cv2.imshow('Enhanced Face Detection - Drishti', display_frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f'detected_faces_frame_{frame_number}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('p'):
            # Pause
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection completed. Total faces detected across all frames: {total_faces_detected}")

def main():
    """
    Main function with enhanced options
    """
    print("Drishti - Enhanced Face Detection")
    print("Uses MediaPipe + YOLO for accurate detection")
    print("=" * 90)
    print("1. Use Camera (Live)")
    print("2. Use Video File")
    print("3. Test with sample images")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        detect_faces(0)
    
    elif choice == '2':
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        video_path = input("Enter the path to your video file: ").strip().strip('"')
        
        if os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            print("Controls:")
            print("- Press 'q' to quit")
            print("- Press 's' to save current frame")
            print("- Press 'p' to pause")
            detect_faces(video_path)
        else:
            print(f"Error: Video file not found at {video_path}")
    
    elif choice == '3':
        print("Test with sample images")
        print("a. Test images in current directory")
        print("b. Test images in a specific folder")
        print("c. Test a single image file")
        
        sub_choice = input("Enter your choice (a/b/c): ").lower()
        
        if sub_choice == 'a':
            test_images_in_directory(".")
        elif sub_choice == 'b':
            folder_path = input("Enter the folder path (e.g., C:/Users/Public/Pictures): ").strip().strip('"')
            if os.path.exists(folder_path):
                test_images_in_directory(folder_path)
            else:
                print(f"Folder not found: {folder_path}")
        elif sub_choice == 'c':
            image_path = input("Enter the path to your image file: ").strip().strip('"')
            if os.path.exists(image_path):
                test_single_image(image_path)
            else:
                print(f"Image file not found: {image_path}")
        else:
            print("Invalid choice.")
    
    else:
        print("Invalid choice. Please run the program again.")

def test_images_in_directory(directory_path):
    """Test face detection on images in a specific directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Get all image files in the directory
    images = []
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory_path, file))
    
    if not images:
        print(f"No image files found in directory: {directory_path}")
        return
    
    print(f"Found {len(images)} images in {directory_path}")
    
    for img_path in images:
        print(f"\nTesting {os.path.basename(img_path)}...")
        test_single_image(img_path)

def test_single_image(image_path):
    """Test face detection on a single image"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return
    
    original_frame = frame.copy()
    
    # Initialize detection models
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    )
    
    print("Loading YOLO model...")
    yolo_model = download_yolo_model()
    
    # Method 1: MediaPipe Detection
    mp_faces = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = mp_face_detection.process(rgb_frame)
    
    if mp_results.detections:
        for detection in mp_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            mp_faces.append(bbox)
            
            # Draw MediaPipe detection in green
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            cv2.putText(frame, f'MP: {detection.score[0]:.2f}', 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Method 2: YOLO Detection
    yolo_faces = []
    if yolo_model is not None:
        try:
            results = yolo_model(frame, conf=0.25, classes=[0])
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        yolo_faces.append(bbox)
                        
                        # Draw YOLO detection in blue
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f'YOLO: {conf:.2f}', 
                                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            print(f"YOLO detection error: {e}")
    

    # Count total faces
    total_faces = len(mp_faces) + len(yolo_faces)
    
    print(f"Results: {total_faces} faces detected (MP: {len(mp_faces)}, YOLO: {len(yolo_faces)})")
    
    # Add text info on image
    cv2.putText(frame, f'Total Faces: {total_faces}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'MP: {len(mp_faces)} | YOLO: {len(yolo_faces)}', 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the image
    window_name = f'Face Detection - {os.path.basename(image_path)}'
    
    # Resize image for display if it's too large
    display_frame = frame.copy()
    max_display_width = 1280
    max_display_height = 720
    
    h, w = frame.shape[:2]
    if w > max_display_width or h > max_display_height:
        scale_w = max_display_width / w
        scale_h = max_display_height / h
        scale = min(scale_w, scale_h)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        display_frame = cv2.resize(display_frame, (new_width, new_height))
        print(f"Image resized for display: {w}x{h} -> {new_width}x{new_height}")
    
    cv2.imshow(window_name, display_frame)
    
    print("Press 's' to save result, any other key to continue...")
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('s'):
        # Save the result
        filename = f'detected_{os.path.basename(image_path)}'
        cv2.imwrite(filename, frame)
        print(f"Result saved as {filename}")
    
    cv2.destroyAllWindows()

def test_images():
    """Legacy function - redirects to test_images_in_directory"""
    test_images_in_directory(".")

if __name__ == "__main__":
    main()
