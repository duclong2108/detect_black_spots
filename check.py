import cv2
import os
import matplotlib.pyplot as plt

# Input and output folder paths
input_folder = './Test'  # Input folder path, replace it with your own path
output_folder = './Output'  # Output folder path, replace it with your own path

def plotImg(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

# Function to delete all files in the output folder
def clear_output_folder():
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    # If the output folder already exists, delete its contents
    clear_output_folder()

def detect_black_spots(img):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to create a binary mask for the object
    _, binary_mask = cv2.threshold(
        gray_image, 50, 255, cv2.THRESH_BINARY
    )
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Create a binary image with adaptive thresholding
    binary_img = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 75, 6.5
    )
    # Ensure that at least one contour was found
    if contours:
        # Get the largest contour (assumes the object is the largest)
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the minimum enclosing circle for the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        # Convert coordinates and radius to integers
        center = (int(x), int(y))
        radius = int(radius)
        # Draw the calculated circle on a copy of the original image
        result_image = binary_img.copy()
        cv2.circle(result_image, center, radius + 20, (0, 0, 0), 9)  # Red circle
        if radius - 55 >= 0:
            cv2.circle(result_image, center, radius - 55, (0, 0, 0), 9)  # Red circle 
        # Find connected components (regions) in the binary image
        _, _, boxes, _ = cv2.connectedComponentsWithStats(result_image)
        boxes = boxes[1:]
        filtered_boxes = []
        # Filter out small and non-rectangular regions
        for x, y, w, h, pixels in boxes:
            if pixels < 10000 and h < 200 and w < 200 and h > 2 and w > 2:
                filtered_boxes.append((x, y, w, h))
                cv2.rectangle(img, (x - 100, y + 100),
                               (x + w + 100, y + h - 100),
                               (255, 255, 255), 2
                               )
        # If there are filtered regions, consider the image to have black spots
        if len(filtered_boxes) > 0:
            return True
        else:
            return False
    else:
        return False

# Loop through all image files in the input folder
for filename in os.listdir(input_folder):
    
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load the image
        img = cv2.imread(input_path)

        if img is not None:
            # Check if the image has black spots
            if detect_black_spots(img):
                # Save the image to the output folder
                cv2.imwrite(output_path, img)

print("Processing completed.")
