# Define the state schema
import os
import numpy as np

from pydantic import BaseModel

from typing import Dict, Any, List, Optional

from matplotlib import pyplot as plt

from ultralytics import YOLO

from langchain_anthropic import ChatAnthropic

from compute import full_image_processing_pipeline, read_image_from_s3

from weather import WeatherData

class LocationModel(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class InputStateModel(BaseModel):
    user_id: Optional[str] = None
    image_key: str
    coordinates: Optional[LocationModel] = None

class WeatherDataOutputModel(BaseModel):
    location: Optional[str] = None
    past_7_days: List[str] = []
    current: Dict[str, Any] = {}
    forecast_next_days: List[str] = []

class YoloAnalysisOutputModel(BaseModel):
    status: Optional[str] = None
    class_labels: List[str] = []
    scores: List[float] = []
    bounding_boxes: List[List[float]] = []
    save_path: Optional[str] = None

class NDVIOutputModel(BaseModel):
    ndvi_summary: Optional[Dict[str, float]] = None
    save_path: Optional[str] = None

class MauiRecommendationModel(BaseModel):
    risk_level: str = "Unknown"
    advice: str = "No advice available"
    data_summary: Dict[str, Any] = {}

class OutputStateModel(BaseModel):
    user_id: Optional[str] = None
    yolo_result: Optional[YoloAnalysisOutputModel] = None
    weather_data: Optional[WeatherDataOutputModel] = None
    ndvi_result: Optional[NDVIOutputModel] = None
    recommendation: Optional[MauiRecommendationModel] = None

# — define the one “State” model that has *all* fields —
class State(InputStateModel, OutputStateModel):
    pass


def check_for_tiff(input_state: State):
    if input_state.image_key.endswith(".tiff"):
        return "NDVI_pipeline"
    else:
        return "YOLO_analysis"



def weather_node(input_state: State) -> WeatherDataOutputModel:
    weather_state = WeatherDataOutputModel()
    latitude = input_state.coordinates.latitude
    longitude = input_state.coordinates.longitude

    if latitude is not None and longitude is not None:
        weather = WeatherData(latitude=latitude, longitude=longitude)
        weather.fetch()  # this populates weather.weather

        if weather.weather:  # make sure it's populated
            weather_state = WeatherDataOutputModel(**weather.weather)

    return {"user_id": input_state.user_id, "weather_data": weather_state.model_dump()}


def YOLO_analysis(input_state: State) -> YoloAnalysisOutputModel:
    image_path = input_state.image_key
    image = read_image_from_s3("qijaniproductsbucket", image_path)
    yolo_state = YoloAnalysisOutputModel()

    try:
        model = YOLO("yolo11s-pest-detection/best.pt")
        results = model.predict(image, save=True)

        if not results:
            yolo_state = {"status": "No detections"}
            return yolo_state

        first_result = results[0]
        yolo_state = {
            "status": "Success",
            "class_labels": first_result.names.values(),
            "scores": [float(c) for c in first_result.boxes.conf.tolist()],
            "bounding_boxes": [b.tolist() for b in first_result.boxes.xyxy],
            "save_path": first_result.save_dir,
        }
        yolo_output = YoloAnalysisOutputModel(
            status=yolo_state["status"],
            class_labels=list(yolo_state["class_labels"]),  # convert dict_values to list
            scores=yolo_state["scores"],
            bounding_boxes=yolo_state["bounding_boxes"],
            save_path=yolo_state["save_path"]
        )

    except Exception as e:
        yolo_state = {"status": f"Error: {str(e)}"}

    return {"user_id": input_state.user_id, "yolo_result": yolo_output.model_dump()}



def NDVI_analysis(input_state: State) -> NDVIOutputModel:
    RADIOMETRIC_PARAMS = {
        'gain': [0.012] * 5,
        'offset': [0] * 5,
        'sunelev': 60.0,
        'edist': 1.0,
        'Esun': [1913, 1822, 1557, 1317, 1074],
        'blackadjust': 0.01,
        'low_percentile': 1
    }
    NOISE_METHOD = 'median'
    NOISE_KERNEL_SIZE = 3
    SIGMA = 1.0

    ndvi_state = NDVIOutputModel()

    try:
        image_key = input_state.image_key
        bucket_name = "qijaniproductsbucket"
        red_band_index = 2
        nir_band_index = 4

        image = read_image_from_s3(bucket_name, image_key)

        ndvi_noise_reduced, _ = full_image_processing_pipeline(
            image,
            RADIOMETRIC_PARAMS,
            detector_type='ORB',
            noise_method=NOISE_METHOD,
            noise_kernel_size=NOISE_KERNEL_SIZE,
            sigma=SIGMA,
            nir_band_index=nir_band_index,
            red_band_index=red_band_index,
            visualize=False,
            use_parallel_noise_reduction=False
        )

        # Save image preview
        image_save_path = f"outputs/ndvi_{os.path.basename(image_key)}.jpg"
        plt.figure(figsize=(10, 8))
        plt.imshow(ndvi_noise_reduced, cmap='RdYlGn')
        plt.colorbar(label='NDVI Value')
        plt.title("NDVI Analysis")
        plt.savefig(image_save_path, dpi=300)
        plt.close()

        # Save NDVI as .npy
        npy_save_path = image_save_path.replace('.jpg', '.npy')
        np.save(npy_save_path, ndvi_noise_reduced)

        ndvi_summary = {
            "min": float(np.min(ndvi_noise_reduced)),
            "max": float(np.max(ndvi_noise_reduced)),
            "mean": float(np.mean(ndvi_noise_reduced))
        }

        ndvi_state = {
            "ndvi_summary": ndvi_summary,
            "save_path": npy_save_path
        }

        ndvi_output = NDVIOutputModel(
            ndvi_summary=ndvi_state["ndvi_summary"],
            save_path=npy_save_path
        )

    except Exception as e:
        ndvi_state = {"error": str(e)}

    return {"user_id": input_state.user_id, "ndvi_result": ndvi_output.model_dump()}


def Maui(input_state: State) -> MauiRecommendationModel:
    model = ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=0.3,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    ndvi_result = getattr(input_state, 'ndvi_result', None)
    ndvi_summary = getattr(ndvi_result, 'ndvi_summary', {'mean': 'No NDVI data'}) if ndvi_result else {'mean': 'No NDVI data'}
    ndvi_image = getattr(ndvi_result, 'save_path', 'No NDVI image saved') if ndvi_result else 'No NDVI image saved'

    yolo_detection = getattr(input_state, 'yolo_result', None)
    weather_summary = getattr(input_state, 'weather_data', {'summary': 'No weather data'})

    context = f"""
You are an AI assistant for precision farmers.

Here is the collected data:

- Pest Detection: {', '.join([f"{label} ({score:.2f})" for label, score in zip(yolo_detection.class_labels, yolo_detection.scores)]) if yolo_detection else 'No pests detected'}
- NDVI Summary: {ndvi_summary}
- NDVI Image Path: {ndvi_image}
- Weather Current Summary: {weather_summary}

Give me:
1. Risk Level: Low / Moderate / High.
2. Advice: Plain actionable language, no jargon.
3. Data Summary: Recap these findings in under 60 words.
"""

    response = model.invoke([{"role": "user", "content": context}])
    response_text = response.content.strip()

    return {"user_id": input_state.user_id, "recommendation": response_text}