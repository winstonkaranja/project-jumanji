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
    latitude: Optional[float]
    longitude: Optional[float]

class InputStateModel(BaseModel):
    user_id: Optional[str]
    image_key: str
    coordinates: LocationModel

class WeatherDataOutputModel(BaseModel):
    location: Optional[str]
    past_7_days: List[str]
    current: Dict[str, Any]
    forecast_next_days: List[str]

class YoloAnalysisOutputModel(BaseModel):
    status: Optional[str]
    class_labels: List[str]
    scores: List[float]
    bounding_boxes: List[List[float]]
    save_path: Optional[str]

class NDVIOutputModel(BaseModel):
    ndvi_summary: Optional[Dict[str, float]]
    save_path: Optional[str]

class MauiRecommendationModel(BaseModel):
    risk_level: str
    advice: str
    data_summary: Dict[str, Any]

class OutputStateModel(BaseModel):
    user_id: Optional[str]
    yolo_result: Optional[YoloAnalysisOutputModel]
    weather_data: Optional[WeatherDataOutputModel]
    ndvi_result: Optional[NDVIOutputModel]
    recommendation: Optional[MauiRecommendationModel]

# — define the one “State” model that has *all* fields —
class State(InputStateModel, OutputStateModel):
    pass


def check_for_tiff(input_state: InputStateModel):
    if input_state["image_key"].endswith(".tiff"):
        return "NDVI_pipeline"
    else:
        return "YOLO_analysis"



def weather_node(input_state: InputStateModel) -> OutputStateModel:
    latitude = input_state.coordinates.latitude
    longitude = input_state.coordinates.longitude
    output_state: OutputStateModel = {"user_id": input_state.user_id}

    if latitude is not None and longitude is not None:
        weather = WeatherData(latitude=latitude, longitude=longitude)
        weather.fetch()
        output_state["weather_data"] = weather.weather

    return output_state

def YOLO_analysis(input_state: InputStateModel) -> OutputStateModel:
    image_path = input_state.image_key
    output_state: OutputStateModel = {"user_id": input_state.user_id}

    try:
        model = YOLO("yolo11s-pest-detection/best.pt")
        results = model.predict(image_path, save=True)

        if not results:
            output_state["yolo_result"] = {"status": "No detections"}
            return output_state

        first_result = results[0]
        output_state["yolo_result"] = {
            "status": "Success",
            "class_labels": first_result.names.values(),
            "scores": [float(c) for c in first_result.boxes.conf.tolist()],
            "bounding_boxes": [b.tolist() for b in first_result.boxes.xyxy],
            "save_path": first_result.save_dir,
        }

    except Exception as e:
        output_state["yolo_result"] = {"status": f"Error: {str(e)}"}

    return output_state



def NDVI_analysis(input_state: InputStateModel) -> OutputStateModel:
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

    output_state: OutputStateModel = {"user_id": input_state.user_id}

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

        output_state["ndvi_result"] = {
            "ndvi_summary": ndvi_summary,
            "save_path": npy_save_path
        }

    except Exception as e:
        output_state.ndvi_result = {"error": str(e)}

    return output_state


def Maui(input_state: OutputStateModel) -> MauiRecommendationModel:
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

- Pest Detection: {', '.join(yolo_detection)}
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

    return {
        "risk_level": "Pending LLM extraction",
        "advice": response_text,
        "data_summary": {
            "pests": yolo_detection,
            "ndvi": ndvi_summary,
            "weather": weather_summary,
            "ndvi_image_path": ndvi_image
        }
    }