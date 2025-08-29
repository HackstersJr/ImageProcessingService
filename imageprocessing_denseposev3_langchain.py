
# FastAPI service for DensePose + Gemini Vision injury detection
import os
import io
import json
import base64
import urllib.request
import requests
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

# Set up API key for Gemini
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "<YOUR_API_KEY>")

app = FastAPI(title="DensePose Injury Detection Service")

# Download configs/weights if not present
os.makedirs("densepose_cfg", exist_ok=True)
cfg_base_url = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs"
cfg_main = "densepose_rcnn_R_50_FPN_s1x.yaml"
cfg_base = "Base-DensePose-RCNN-FPN.yaml"

def fetch(url, out):
    if not os.path.exists(out):
        print("Downloading", out)
        urllib.request.urlretrieve(url, out)

fetch(f"{cfg_base_url}/{cfg_main}", f"densepose_cfg/{cfg_main}")
fetch(f"{cfg_base_url}/{cfg_base}", f"densepose_cfg/{cfg_base}")
weights_url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
weights_path = "model_final_162be9.pkl"
fetch(weights_url, weights_path)

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor

cfg_path = "densepose_cfg/densepose_rcnn_R_50_FPN_s1x.yaml"
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(cfg_path)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.INPUT.MIN_SIZE_TEST = 800
predictor = DefaultPredictor(cfg)

COARSE_GROUPS = {
    "head":   {23, 24},
    "torso":  {1, 2},
    "left_arm":  {16, 18, 20, 22, 4},
    "right_arm": {15, 17, 19, 21, 3},
    "left_leg":  {8, 10, 12, 14, 6},
    "right_leg": {7,  9, 11, 13, 5},
}

# Gemini Vision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def classify_injury(image_path: str):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    msg = HumanMessage(
        content=[
            {"type": "text", "text":
             "You are a medical triage assistant. Analyze this human body part image and answer in strict JSON:\n"
             "{ \"injury\": true/false, \"confidence\": float (0-1), \"notes\": string }\n"
             "Detect wounds, bleeding, burns, fractures, or visible trauma. "
             "If none, return { \"injury\": false, \"confidence\": ~0.8-0.95, \"notes\": \"No injury detected\" }."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]
    )
    resp = vision_llm.invoke([msg])
    try:
        result = json.loads(resp.content)
    except Exception:
        result = {"injury": False, "confidence": 0.0, "notes": resp.content}
    return result

def process_image(image_bytes, save_dir="segmented_parts"):
    os.makedirs(save_dir, exist_ok=True)
    im_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise RuntimeError("Failed to decode image.")
    H, W = im_bgr.shape[:2]
    with torch.no_grad():
        outputs = predictor(im_bgr)
    instances = outputs["instances"].to("cpu")
    people_json = []
    full_I_mask = np.zeros((H, W), dtype=np.uint8)
    if len(instances) == 0 or not instances.has("pred_densepose"):
        return {
            "people": [],
            "message": "No people or DensePose results detected."
        }, im_bgr, people_json
    scores = instances.scores.tolist() if instances.has("scores") else [1.0] * len(instances)
    extractor = DensePoseResultExtractor()
    dp_results, boxes_xywh = extractor(instances)
    for i, (dp_res, box_xywh, score) in enumerate(zip(dp_results, boxes_xywh, scores)):
        I_roi = dp_res.labels.cpu().numpy().astype(np.uint8)
        x, y, w, h = [int(v) for v in box_xywh.tolist()]
        w = max(w, 1); h = max(h, 1)
        I_resized = cv2.resize(I_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        I_person = np.zeros((H, W), dtype=np.uint8)
        y2 = min(y+h, H); x2 = min(x+w, W)
        I_person[y:y2, x:x2] = I_resized[:(y2-y), :(x2-x)]
        full_I_mask[y:y2, x:x2] = np.where(I_resized[:(y2-y), :(x2-x)]>0,
                                        I_resized[:(y2-y), :(x2-x)],
                                        full_I_mask[y:y2, x:x2])
        parts_present = {}
        saved_parts = {}
        for name, ids in COARSE_GROUPS.items():
            mask_bin = np.isin(I_person, list(ids)).astype(np.uint8)
            if mask_bin.sum() > 0:
                parts_present[name] = int(mask_bin.sum())
                masked = im_bgr.copy()
                mask_bin_3_channel = np.stack([mask_bin, mask_bin, mask_bin], axis=-1)
                masked = masked * mask_bin_3_channel
                ys, xs = np.where(mask_bin > 0)
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                crop = masked[ymin:ymax, xmin:xmax]
                out_path = os.path.join(save_dir, f"person{i}_{name}.jpg")
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path, "JPEG")
                saved_parts[name] = out_path
        person_entry = {
            "bbox_xywh": [x, y, w, h],
            "score": float(score),
            "parts_detected": parts_present,
            "part_files": saved_parts,
        }
        people_json.append(person_entry)
    return {"people": people_json}, im_bgr, people_json

def classify_all_parts(people_json, save_dir="segmented_parts"):
    injury_results = {}
    for i, person_data in enumerate(people_json):
        person_injury_results = {}
        for part_name, part_file in person_data["part_files"].items():
            injury_info = classify_injury(part_file)
            person_injury_results[part_name] = injury_info
        injury_results[f"person_{i}"] = person_injury_results
    output_injury_json_path = os.path.join(save_dir, "injury_classification_results.json")
    with open(output_injury_json_path, "w") as f:
        json.dump(injury_results, f, indent=2)
    return injury_results

class ProcessRequest(BaseModel):
    image_url: Optional[str] = None
    post_api_url: Optional[str] = None

@app.post("/process")
async def process_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    post_api_url: Optional[str] = Form(None)
):
    """
    Accepts an uploaded image or image_url, processes it, and posts results to post_api_url if provided.
    """
    if file:
        image_bytes = await file.read()
    elif image_url:
        resp = requests.get(image_url)
        if resp.status_code != 200:
            return JSONResponse(status_code=400, content={"error": "Failed to fetch image from URL."})
        image_bytes = resp.content
    else:
        return JSONResponse(status_code=400, content={"error": "No image provided."})

    payload, im_bgr, people_json = process_image(image_bytes)
    injury_results = classify_all_parts(people_json)
    result = {
        "densepose": payload,
        "injury_results": injury_results
    }
    # Optionally post to another API
    if post_api_url:
        try:
            post_resp = requests.post(post_api_url, json=result)
            result["post_status"] = post_resp.status_code
        except Exception as e:
            result["post_status"] = f"Failed: {e}"
    return result

# Health check
@app.get("/")
def root():
    return {"status": "ok"}

import os

# put your API key here securely
os.environ["GOOGLE_API_KEY"] = "AIzaSyC4NqPBPBUjGBOd5Hlbz48-bGlFoFDnm_s"

import os, urllib.request

os.makedirs("densepose_cfg", exist_ok=True)

# Grab canonical config files from Detectron2's DensePose project (raw GitHub)
cfg_base_url = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs"
cfg_main = "densepose_rcnn_R_50_FPN_s1x.yaml"
cfg_base = "Base-DensePose-RCNN-FPN.yaml" # Add base config

def fetch(url, out):
    if not os.path.exists(out):
        print("Downloading", out)
        urllib.request.urlretrieve(url, out)

# Main model config
fetch(f"{cfg_base_url}/{cfg_main}", f"densepose_cfg/{cfg_main}")
# Base model config
fetch(f"{cfg_base_url}/{cfg_base}", f"densepose_cfg/{cfg_base}")


# Model weights (R50-FPN s1x model used in DensePose docs)
# If this ever 404s, open the URL in a browser and copy the redirected link you get.
weights_url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
weights_path = "model_final_162be9.pkl"
fetch(weights_url, weights_path)

print("Config & weights ready:", os.path.exists(f"densepose_cfg/{cfg_main}"), os.path.exists(weights_path))

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config  # provided by the DensePose package

cfg_path = "/content/densepose_cfg/densepose_rcnn_R_50_FPN_s1x.yaml"

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(cfg_path)
cfg.MODEL.WEIGHTS = "model_final_162be9.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # tweak per your needs
cfg.INPUT.MIN_SIZE_TEST = 800

# GPU if available, otherwise CPU (slower)
predictor = DefaultPredictor(cfg)

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# Option A: Upload from your machine
from google.colab import files
uploaded = files.upload()  # pick one image
IMAGE_PATH = list(uploaded.keys())[0]

# --- Option B (instead): URL fetch ---
# import urllib.request
# IMAGE_PATH = "test.jpg"
# urllib.request.urlretrieve("https://path/to/your/person.jpg", IMAGE_PATH)

print("Using:", IMAGE_PATH)

import cv2, json, base64, io
import numpy as np
from PIL import Image
import torch
import os

# DensePose result extractor (handles Instances -> DensePoseResult + boxes)
from densepose.vis.extractor import DensePoseResultExtractor

# Coarse grouping of DensePose 24 part indices to limb labels (I-channel).
# Mapping is based on DensePose part index conventions discussed by the authors/users:
# 0=bg; 1,2=torso; 3/4=hands; 5/6=feet; 7-14=upper/lower legs (R/L); 15-22=upper/lower arms (R/L); 23=head; 24=neck.
# (DensePose site/papers explain the IUV format; original repo is archived; see docs.)
# We aggregate these into head/torso/left_arm/right_arm/left_leg/right_leg.
COARSE_GROUPS = {
    "head":   {23, 24},
    "torso":  {1, 2},
    "left_arm":  {16, 18, 20, 22, 4},   # upper L arm, lower L arm, left hand
    "right_arm": {15, 17, 19, 21, 3},   # upper R arm, lower R arm, right hand
    "left_leg":  {8, 10, 12, 14, 6},    # upper L leg, lower L leg, left foot
    "right_leg": {7,  9, 11, 13, 5},    # upper R leg, lower R leg, right foot
}


# Read image
im_bgr = cv2.imread(IMAGE_PATH)
if im_bgr is None:
    raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")
H, W = im_bgr.shape[:2]

# Inference
with torch.no_grad():
    outputs = predictor(im_bgr)

instances = outputs["instances"].to("cpu")

# Check if any instances (people) were detected and if DensePose results exist
if len(instances) == 0 or not instances.has("pred_densepose"):
    print("No people or DensePose results detected. Saving original image.")
    # Save the original image
    output_image_path = "output_image.jpg"
    Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)).save(output_image_path, "JPEG")
    print(f"Original image saved as {output_image_path}")

    # Create an empty JSON payload
    payload = {
        "source_image": IMAGE_PATH,
        "people": [],
        "densepose_model": {
            "config": os.path.basename(cfg_path),
            "weights": os.path.basename(weights_path),
            "framework": "detectron2 + densepose",
        },
        "message": "No people or DensePose results detected. Original image saved."
    }
    out_json = "densepose_singleframe.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote empty {out_json}")

else:
    scores = instances.scores.tolist() if instances.has("scores") else [1.0] * len(instances)

    # Extract DensePose results + boxes (xywh)
    extractor = DensePoseResultExtractor()
    dp_results, boxes_xywh = extractor(instances)


    os.makedirs("segmented_parts", exist_ok=True)

    people_json = []
    full_I_mask = np.zeros((H, W), dtype=np.uint8)  # optional composite mask

    for i, (dp_res, box_xywh, score) in enumerate(zip(dp_results, boxes_xywh, scores)):
        # dp_res.labels: [H_roi, W_roi], values in [0..24]; dp_res.uv: (2, H_roi, W_roi) but not needed for coarse parts.
        I_roi = dp_res.labels.cpu().numpy().astype(np.uint8)

        # Paste ROI labels into image coords using bbox (nearest-neighbor)
        x, y, w, h = [int(v) for v in box_xywh.tolist()]
        w = max(w, 1); h = max(h, 1)
        I_resized = cv2.resize(I_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        # Compose per-person mask (same size as image; zero elsewhere)
        I_person = np.zeros((H, W), dtype=np.uint8)
        y2 = min(y+h, H); x2 = min(x+w, W)
        I_person[y:y2, x:x2] = I_resized[:(y2-y), :(x2-x)]

        # Update composite
        full_I_mask[y:y2, x:x2] = np.where(I_resized[:(y2-y), :(x2-x)]>0,
                                        I_resized[:(y2-y), :(x2-x)],
                                        full_I_mask[y:y2, x:x2])

        # Count pixels per coarse group and save masks
        parts_present = {}
        saved_parts = {}

        for name, ids in COARSE_GROUPS.items():
            # Extract only if present
            mask_bin = np.isin(I_person, list(ids)).astype(np.uint8)

            if mask_bin.sum() > 0:
                parts_present[name] = int(mask_bin.sum())

                # Apply mask on original image
                masked = im_bgr.copy()
                # Create a 3-channel mask from the single-channel binary mask
                mask_bin_3_channel = np.stack([mask_bin, mask_bin, mask_bin], axis=-1)
                masked = masked * mask_bin_3_channel # Apply the mask

                # Crop tight region of detected part
                ys, xs = np.where(mask_bin > 0)
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                crop = masked[ymin:ymax, xmin:xmax]

                # Save as JPG
                out_path = f"segmented_parts/person{i}_{name}.jpg"
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path, "JPEG")
                saved_parts[name] = out_path


        person_entry = {
            "bbox_xywh": [x, y, w, h],
            "score": float(score),
            "parts_detected": parts_present,
            "part_files": saved_parts,  # file paths for cropped regions
        }
        people_json.append(person_entry)

    # Pack final JSON
    payload = {
        "source_image": IMAGE_PATH,
        "people": people_json,
        "densepose_model": {
            "config": os.path.basename(cfg_path),
            "weights": os.path.basename(weights_path),
            "framework": "detectron2 + densepose",
        },
    }

    out_json = "densepose_singleframe.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2) # Use indent=2 for pretty printing the JSON

    print(f"Wrote {out_json} with {len(people_json)} person(s). Segmented parts in segmented_parts/")


    import matplotlib.pyplot as plt

    # Simple 25-color palette (0..24) for visualization
    palette = np.random.RandomState(123).randint(0, 255, (25, 3), dtype=np.uint8)
    palette[0] = 0  # background black

    # Build overlay for the composite mask
    color_mask = palette[full_I_mask]
    overlay = (0.6 * cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) + 0.4 * color_mask).astype(np.uint8)

    # Draw bboxes
    for p in payload["people"]:
        x, y, w, h = p["bbox_xywh"]
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)

    plt.figure(figsize=(10, 10))
    plt.title("DensePose parts overlay (with person boxes)")
    plt.axis("off")
    plt.imshow(overlay)
    plt.show()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
import json # Import json here as well for the fallback

# Initialize Gemini Vision model
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def classify_injury(image_path: str):
    """
    Sends cropped part image to Gemini Vision for injury detection.
    Returns dict {injury: true/false, confidence: float, notes: str}.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Construct multimodal input
    msg = HumanMessage(
        content=[
            {"type": "text", "text":
             "You are a medical triage assistant. Analyze this human body part image and answer in strict JSON:\n"
             "{ \"injury\": true/false, \"confidence\": float (0-1), \"notes\": string }\n"
             "Detect wounds, bleeding, burns, fractures, or visible trauma. "
             "If none, return { \"injury\": false, \"confidence\": ~0.8-0.95, \"notes\": \"No injury detected\" }."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]
    )

    resp = vision_llm.invoke([msg])
    # Parse out JSON safely
    try:
        result = json.loads(resp.content)
    except Exception:
        # fallback if not JSON
        result = {"injury": False, "confidence": 0.0, "notes": resp.content}
    return result



# Iterate through detected people and their parts to classify injuries
injury_results = {}

if len(payload["people"]) > 0:
    for i, person_data in enumerate(payload["people"]):
        person_injury_results = {}
        for part_name, part_file in person_data["part_files"].items():
            print(f"Classifying injury for person {i}, {part_name}...")
            injury_info = classify_injury(part_file)
            person_injury_results[part_name] = injury_info
            print(f"  Result: {injury_info}")
        injury_results[f"person_{i}"] = person_injury_results
else:
    # Handle the case where no people were detected.
    # The previous cell already saved the original image and created a JSON payload.
    # We can add a placeholder in the injury results indicating no people were found.
    injury_results["message"] = payload["message"]
    print("No people detected, no injury classification performed.")


# You can now access the injury_results dictionary to see the classification for each part.
# For example, to see the results for the first person's head:
# print(injury_results["person_0"]["head"])

# Optionally, save the injury results to a JSON file in the segmented_parts folder
import json
output_injury_json_path = "segmented_parts/injury_classification_results.json"
with open(output_injury_json_path, "w") as f:
    json.dump(injury_results, f, indent=2)

print(f"Injury classification results saved to {output_injury_json_path}")

