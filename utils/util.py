from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
import datetime as dt
import time

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days=366)
    return day + dayfrac


def extract_face(file_name:str, output_size:tuple=None) -> None:
        
	# Load image
    img = Image.open(file_name)
    img_array = np.asarray(img)
    
    # Instantiate detector
    face_detector = MTCNN()

	# Detect face
    results = face_detector.detect_faces(img_array)
    x1, y1, w, h = results[0]['box']
    x2 = x1 + w
    y2 = y1 + h

	# Extract face
    face = img_array[y1:y2, x1:x2]

	# Resize to shape
    if output_size is not None:
        image = Image.fromarray(face)
        image = image.resize(output_size)
        trans_face = np.asarray(image)
        return trans_face
    else:
        return face

def extract_faces(path_images:str, path_cropped_images:str=None, output_image_size:tuple=None) -> dict:
    # Create dict for holding images
    faces = {}

    # Confirm path to store images to
    os.makedirs(path_cropped_images, exist_ok=True)
    
    # Extract faces
    for file_name in os.listdir(path_images):
        file_path = os.path.join(path_images, file_name)

        # Extract face
        face = extract_face(file_path, output_image_size)

        # Add face to dict
        faces[file_name] = face

        # Save image to folder
        if path_cropped_images is not None:
            img = Image.fromarray(face)
            img.save(os.path.join(path_cropped_images, file_name))

def extract_faces_paths(root_path:str, original_paths:list, path_cropped_images:str=None, output_image_size:tuple=None) -> dict:
    # Confirm path to store images to
    os.makedirs(path_cropped_images, exist_ok=True)

    counter = 0
    total = len(original_paths)
    logged_times = []
    
    # Extract faces
    for file_name in original_paths:
        
        start = time.time()
        file_path = os.path.join(root_path, file_name)

        new_path = os.path.join(path_cropped_images, file_name)
        
        counter += 1

        if os.path.exists(new_path):
            continue

        # Extract face
        try:
            face = extract_face(file_path, output_image_size)
        except:
            continue
        
        try:
            # Save image to folder
            if path_cropped_images is not None:
                
                os.makedirs(new_path[:-len(new_path.split("/")[-1])], exist_ok=True)

                img = Image.fromarray(face)
                img.save(new_path)

                # if counter % 10 == 9:
                elapsed_time = time.time() - start
                logged_times.append(elapsed_time)
                
                if len(logged_times) > 50:
                    logged_times = logged_times[-50:]
                
                    print(f"{counter}/{total} - ETA: {time.strftime('%H:%M:%S', time.gmtime(np.mean(logged_times) * (total - counter)))}")
        except:
            continue