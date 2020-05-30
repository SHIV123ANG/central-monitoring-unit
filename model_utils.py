import os
import json
import time
import numpy as np

from face_encoder.encoder import encode_face as ef
from face_detector.detector import detect_faces as dfs
from config import (
    FACE_MATCH_THRESHOLD, 
    FACE_IMAGE_PATH, 
    PERSON_DATA_PATH
)

def encode_face(img):
    '''
    img: numpy array of shape HxWx3 and data type uint8.
    Returns: N-dimensional face encoding.
    '''
    return ef(img)


def encodings_cosine(enc1, enc2):
    '''
    enc1: encoding of first face (np.array).
    enc2: encoding of second face (np.array).
    Both encodings should be of same dimension.
    Returns: True if both faces are same otherwise False.
    '''
    enc1 = enc1/np.linalg.norm(enc1)
    enc2 = enc2/np.linalg.norm(enc2)
    cosine = np.sum(enc1*enc2)
    return cosine


def detect_faces(img):
    '''
    img: numpy array of shape HxWx3 and data type uint8.
    Returns a list of bounding boxes of format (x1, y1, x2, y2).
    '''
    return dfs(img)


def crop_face(img, bbox):
    '''
    img: numpy array of shape HxWx3 and data type uint8.
    bbox: bounding box of format (x1, y1, x2, y2).
    Returns: cropped images] of format numpy array of shape HxWx3 and data type uint8.
    '''
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    return face


def search_matching_faces(img, bboxes, token, progress_dict):
    '''
    img: numpy array of shape HxWx3 and data type uint8.
    bboxes: list of bounding boxes of format (x1, y1, x2, y2).
    token: unique token to track progress of task and get result.
    progress_dict: progress_dict mapped to token as key and task progress as value.
    Returns: list of matched face id corresponding to each bbox.
    '''
    results = []
    prev_progress = 0.0
    person_data_list = os.listdir(PERSON_DATA_PATH)
    num_persons = len(person_data_list)
    num_bboxes = len(bboxes)
    for i in range(num_bboxes):
        bbox = bboxes[i]
        face = crop_face(img, bbox)
        face_encoding = encode_face(face)
        matched_faces = []
        cosine_dists = []
        result = {
            "bbox": bbox, 
            "matched_faces": []
        }
        for j in range(num_persons):
            person_data = os.path.join(PERSON_DATA_PATH, person_data_list[j])
            with open(person_data, "r") as f:
                person_data = json.loads(f.read())
            person_id = person_data["id"]
            person_face_encoding = np.array(person_data["face_encoding"])
            cosine = encodings_cosine(face_encoding, person_face_encoding)
            if cosine >= FACE_MATCH_THRESHOLD:
                matched_faces.append(person_id)
                cosine_dists.append(cosine)
            new_progress = ((i+1)*(j+1))/(num_bboxes*num_persons)
            if(new_progress - prev_progress >= 0.01):
                progress_dict[token] = new_progress
                prev_progress = new_progress
        sorted_indices = np.argsort(cosine_dists)[::-1]
        sorted_matched_faces = matched_faces.copy()
        for ind in range(len(matched_faces)):
            sorted_matched_faces[ind] = matched_faces[sorted_indices[ind]]
        result["matched_faces"] = sorted_matched_faces
        results.append(result)
    progress_dict[token] = 1.0
    return results
