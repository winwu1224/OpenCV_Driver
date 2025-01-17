import cv2
import numpy as np

def calculate_brightness(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the average brightness
    mean_brightness = np.mean(gray_frame)
    return mean_brightness

def adaptive_threshold(mean_brightness, base_threshold=50):
    # Define the maximum and minimum threshold values
    max_threshold = 255
    min_threshold = 0
    
    # Compute a dynamic threshold based on the average brightness
    # For brighter images, increase the threshold, for darker images, decrease it
    adaptive_threshold_value = base_threshold + (128 - mean_brightness) * (base_threshold / 128)
    adaptive_threshold_value = np.clip(adaptive_threshold_value, min_threshold, max_threshold)
    
    return int(adaptive_threshold_value)

def apply_dark_threshold(frame, darkness_threshold):
    # Convert the entire frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a fixed darkness threshold to create a binary mask
    _, thresholded_frame = cv2.threshold(gray_frame, darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_frame

def remove_noise(binary_frame, kernel_size=(3, 3), erosion_iterations=1, dilation_iterations=1):
    # Define kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    
    # Apply erosion to remove small noise points
    eroded_frame = cv2.erode(binary_frame, kernel, iterations=erosion_iterations)
    
    # Apply dilation to fill in small holes
    cleaned_frame = cv2.dilate(eroded_frame, kernel, iterations=dilation_iterations)
    
    return cleaned_frame

def draw_contours(original_frame, binary_frame, min_contour_area):
    # Find contours in the binary frame
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around the detected contours that are larger than min_contour_area
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return original_frame

def color_segmentation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define ranges for black wetsuits and bright neon shirts
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    lower_neon = np.array([20, 100, 100]) # Adjust based on specific neon color ranges
    upper_neon = np.array([40, 255, 255])
    
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_neon = cv2.inRange(hsv, lower_neon, upper_neon)

    # Dilate masks to allow for proximity detection
    kernel = np.ones((5, 5), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=2)
    mask_neon_dilated = cv2.dilate(mask_neon, kernel, iterations=2)
    
    combined_mask = cv2.bitwise_or(mask_black, mask_neon)
    return combined_mask

def process_video(input_path, output_path, debug_path, base_threshold, min_contour_area, kernel_size, erosion_iterations, dilation_iterations):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writers for output and debug videos
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    debug_out = cv2.VideoWriter(debug_path, fourcc, fps, (width, height), isColor=False)  # Black & white video
    
    # Create background subtractor
    # history: The number of frames used to build the background model. 500
    # varThreshold: The threshold for deciding if a pixel is part of the background. 50
    back_sub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75, detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate average brightness of the frame
        mean_brightness = calculate_brightness(frame)
        
        # Determine the adaptive threshold based on brightness
        darkness_threshold = adaptive_threshold(mean_brightness, base_threshold)
        print("using threshold: %.2f" % darkness_threshold)
        
        # # Apply the adaptive darkness threshold
        # thresholded_frame = apply_dark_threshold(frame, darkness_threshold)
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        # Remove noise from the thresholded frame
        cleaned_fg_mask = remove_noise(fg_mask, kernel_size, erosion_iterations, dilation_iterations)

        color_mask = color_segmentation(frame)
        combined_mask = cv2.bitwise_and(cleaned_fg_mask, color_mask)

        cleaned_combined_mask = remove_noise(combined_mask, kernel_size, erosion_iterations, dilation_iterations)

        # Write the black-and-white (cleaned) frame to the debug video
        debug_out.write(cleaned_combined_mask)
        
        # Draw contours on the original frame
        contour_frame = draw_contours(frame.copy(), cleaned_combined_mask, min_contour_area)
        
        # Write the processed frame to the output video
        out.write(contour_frame)
    
    cap.release()
    out.release()
    debug_out.release()
    print("Video processing complete.")

# Paths to the input, output, and debug videos
input_video_path = '/Users/winstonwu/Documents/Surf_Vids/vid1.MP4'
output_video_path = '/Users/winstonwu/Documents/Surf_Vids/output1.mp4'
debug_video_path = '/Users/winstonwu/Documents/Surf_Vids/debug1.mp4'
base_threshold = 50  # Base darkness value to adjust from (0-255)
min_contour_area = 500  # Minimum area of contours to retain
kernel_size = (3, 3)  # Kernel size for morphological operations
erosion_iterations = 1  # Number of erosion iterations
dilation_iterations = 1 # Number of dilation iterations

process_video(input_video_path, output_video_path, debug_video_path, base_threshold, min_contour_area, kernel_size, erosion_iterations, dilation_iterations)
