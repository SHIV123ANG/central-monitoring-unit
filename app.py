import os
import pickle
import json
from flask import (
    Flask, 
    request, 
    jsonify, 
    send_from_directory
)

from flask_jwt_extended import (
    JWTManager, 
    create_access_token, 
    get_jwt_identity, 
    jwt_required, 
    jwt_optional, 
    fresh_jwt_required, 
    verify_jwt_in_request, 
    verify_fresh_jwt_in_request
)

from utils import (
    verify_data, 
    verify_file_extension, 
    read_image, 
    encode_to_base64, 
    decode_from_base64, 
    get_unique_id, 
    get_system_info, 
    get_person_data
)

from model_utils import (
    detect_faces, 
    encode_face, 
    search_matching_faces
)

from concurrency_utils import (
    pool, 
    mp_progress_dict, 
    mp_result_dict, 
    get_progress, 
    get_result, 
    delete_task, 
    get_total_tasks
)

from config import (
    PERSON_DATA_PATH, 
    FACE_IMAGE_PATH, 
    STARRED_PERSON_COUNT_LIMIT
)

from db.utils import (
    get_person_data_from_database, 
    create_connection, 
    search_person_by_name, 
    auth_user, 
    add_star_to_person, 
    remove_star_from_person, 
    get_starred_persons
)

from blacklist import blacklist

with open(os.path.join("docs", "secret_key.txt"), "r") as f:
    secret_key = f.readline()

with open(os.path.join("docs", "api_format.json"), "r") as f:
    api_format = json.load(f)
status_codes = api_format["status_codes"]
response_messages = api_format["response_messages"]

app = Flask(__name__)

app.config["JWT_TOKEN_LOCATION"] = "json"
app.config["JWT_SECRET_KEY"] = secret_key
app.config["JWT_JSON_KEY"] = "access_token" # Key to look for access token in JSON
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 2592000
app.config["JWT_BLACKLIST_ENABLED"] = True # Enable/Disable token revoking
app.config["JWT_BLACKLIST_TOKEN_CHECKS"] = "access" # What token types to check against the blacklist.

jwt = JWTManager(app)

@jwt.unauthorized_loader
def unauthorized_callback(msg):
    resp = {
        "status_code": status_codes["unauthorized_loader"], 
        "message": response_messages["unauthorized_loader"]
    }
    return jsonify(resp)


@jwt.revoked_token_loader
def revoked_token_callback():
    resp = {
        "status_code": status_codes["revoked_token_loader"], 
        "message": response_messages["revoked_token_loader"]
    }
    return jsonify(resp)


@jwt.invalid_token_loader
def invalid_token_callback(msg):
    resp = {
        "status_code": status_codes["invalid_token_loader"], 
        "message": response_messages["invalid_token_loader"]
    }
    return jsonify(resp)


@jwt.expired_token_loader
def expired_token_callback(data_dict):
    resp = {
        "status_code": status_codes["expired_token_loader"], 
        "message": response_messages["expired_token_loader"]
    }
    return jsonify(resp)


@jwt.needs_fresh_token_loader
def needs_fresh_token_callback():
    resp = {
        "status_code": status_codes["needs_fresh_token_loader"], 
        "message": response_messages["needs_fresh_token_loader"]
    }
    return jsonify(resp)


@jwt.token_in_blacklist_loader
def token_in_blacklist_callback(data_dict):
    if data_dict["identity"] in blacklist:
        return True
    return False


@app.route("/login", methods=["POST"])
def login_callback():
    data = request.get_json()
    if not verify_data(data, "username", "password"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    if not auth_user(data["username"], data["password"]):
        resp = {
            "status_code": status_codes["invalid_credentials"], 
            "message": response_messages["invalid_credentials"]
        }
        return jsonify(resp)
    access_token = create_access_token(identity=data["username"], fresh=True)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "access_token": access_token
    }
    return jsonify(resp)


@app.route("/get-fresh-token", methods=["POST"])
@jwt_required
def get_fresh_token_callback():
    data = request.get_json()
    if not verify_data(data, "password"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    identity = get_jwt_identity()
    if not verify_user(identity, data["password"]):
        resp = {
            "status_code": status_codes["invalid_credentials"], 
            "message": response_messages["invalid_credentials"]
        }
        return jsonify(resp)
    access_token = create_access_token(identity=identity, fresh=True)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "access_token": access_token
    }
    return jsonify(resp)


@app.route("/", methods=["POST"])
@jwt_required
def root_callback():
    info = get_system_info()
    total_tasks, completed_tasks = get_total_tasks()
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "total_tasks": total_tasks, 
        "completed_tasks": completed_tasks
    }
    for k, v in info.items():
        resp[k] = v
    return jsonify(resp)

@app.route("/get-face-image", methods=["POST"])
@jwt_required
def get_face_image_callback():
    data = request.get_json()
    if not verify_data(data, "image_name"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    img_name = data["image_name"]
    img_dir = os.path.join(FACE_IMAGE_PATH, img_name)
    if not os.path.exists(img_dir):
        resp = {
            "status_code": status_codes["image_not_found"], 
            "message": response_messages["image_not_found"]
        }
        return jsonify(resp)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"]
    }
    resp["image_data"] = encode_to_base64(img_dir)
    return jsonify(resp)


@app.route("/get-person-data", methods=["POST"])
@jwt_required
def get_person_data_callback():
    data = request.get_json()
    if not verify_data(data, "id"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    data_dir = os.path.join(PERSON_DATA_PATH, data["id"]+".json")
    if not os.path.exists(data_dir):
        resp = {
            "status_code": status_codes["person_not_found"], 
            "message": response_messages["person_not_found"]
        }
        return jsonify(resp)
    d_json = get_person_data(data["id"])
    temp = get_person_data_from_database(data["id"])
    if temp is not None:
        d_json["star"] = temp["star"]
    img_dir = os.path.join(FACE_IMAGE_PATH, d_json["image"])
    d_json["image"] = encode_to_base64(img_dir)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "data": d_json
    }
    return jsonify(resp)


@app.route("/detect-faces", methods=["POST"])
@jwt_required
def detect_faces_callback():
    data = request.get_json()
    if not verify_data(data, "image_data", "image_name"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    if not verify_file_extension(data["image_name"]):
        resp = {
            "status_code": status_codes["file_extension_error"], 
            "message": response_messages["file_extension_error"]
        }
        return jsonify(resp)
    temp_img_name = get_unique_id() + "." + data["image_name"].split(".")[-1]
    temp_img_path = os.path.join("temp", temp_img_name)
    decode_from_base64(data["image_data"], temp_img_path)
    img = read_image(temp_img_path)
    bboxes = detect_faces(img)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "bboxes": bboxes
    }
    return jsonify(resp)


@app.route("/search-matching-faces", methods=["POST"])
@jwt_required
def search_matching_faces_callback():
    data = request.get_json()
    if not verify_data(data, "image_data", "image_name", "bboxes"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    if not verify_file_extension(data["image_name"]):
        resp = {
            "status_code": status_codes["file_extension_error"], 
            "message": response_messages["file_extension_error"]
        }
        return jsonify(resp)
    temp_img_name = get_unique_id() + "." + data["image_name"].split(".")[-1]
    temp_img_path = os.path.join("temp", temp_img_name)
    decode_from_base64(data["image_data"], temp_img_path)
    img = read_image(temp_img_path)
    bboxes = data["bboxes"]
    token = get_unique_id()
    mp_progress_dict[token] = 0.0
    proc = pool.apply_async(search_matching_faces, (img, bboxes, token, mp_progress_dict))
    mp_result_dict[token] = proc
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "token": token
    }
    return jsonify(resp)


@app.route("/get-task-progress", methods=["POST"])
@jwt_required
def get_task_progress_callback():
    data = request.get_json()
    if not verify_data(data, "token"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    progress = get_progress(data["token"])
    if progress is None:
        resp = {
            "status_code": status_codes["task_not_found"], 
            "message": response_messages["task_not_found"]
        }
        return jsonify(resp)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "progress": progress
    }
    return jsonify(resp)


@app.route("/get-task-result", methods=["POST"])
@jwt_required
def get_task_result_callback():
    data = request.get_json()
    if not verify_data(data, "token"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    result = get_result(data["token"])
    if result is None:
        resp = {
            "status_code": status_codes["task_not_found"], 
            "message": response_messages["task_not_found"]
        }
        return jsonify(resp)
    if result is False:
        resp = {
            "status_code": status_codes["task_not_completed"], 
            "message": response_messages["task_not_completed"]
        }
        return jsonify(resp)
    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "result": result
    }
    # if result is not False:
    #     delete_task(data["token"])
    return jsonify(resp)


@app.route("/search-person", methods=["POST"])
@jwt_required
def search_person_callback():
    data = request.get_json()
    if not verify_data(data, "name"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    persons = search_person_by_name(data["name"], limit=100)
    result = []
    for mid, star in persons:
        json_d = get_person_data(mid)
        json_d["star"] = star
        img_dir = os.path.join(FACE_IMAGE_PATH, json_d["image"])
        json_d["image"] = encode_to_base64(img_dir)
        result.append(json_d)

    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "result": result
    }
    return jsonify(resp)

@app.route("/add-star-to-person", methods=["POST"])
@jwt_required
def add_star_to_person_callback():
    data = request.get_json()
    if not verify_data(data, "id"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    result = add_star_to_person(data["id"], STARRED_PERSON_COUNT_LIMIT)
    if not result:
        resp = {
            "status_code": status_codes["starred_persons_limit_reached"], 
            "message": response_messages["starred_persons_limit_reached"]
        }
        return jsonify(resp)
    resp = {
            "status_code": status_codes["success"], 
            "message": response_messages["success"], 
            "result": result
        }
    return jsonify(resp)

@app.route("/remove-star-from-person", methods=["POST"])
@jwt_required
def remove_star_from_person_callback():
    data = request.get_json()
    if not verify_data(data, "id"):
        resp = {
            "status_code": status_codes["insufficient_data"], 
            "message": response_messages["insufficient_data"]
        }
        return jsonify(resp)
    remove_star_from_person(data["id"])
    resp = {
            "status_code": status_codes["success"], 
            "message": response_messages["success"], 
            "result": True
        }
    return jsonify(resp)

@app.route("/get-starred-persons", methods=["POST"])
@jwt_required
def get_starred_persons_callback():
    persons = get_starred_persons()
    result = []
    for mid, star in persons:
        json_d = get_person_data(mid)
        json_d["star"] = star
        img_dir = os.path.join(FACE_IMAGE_PATH, json_d["image"])
        json_d["image"] = encode_to_base64(img_dir)
        result.append(json_d)

    resp = {
        "status_code": status_codes["success"], 
        "message": response_messages["success"], 
        "result": result
    }
    return jsonify(resp)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8080, debug=True)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt. Stopping server")
