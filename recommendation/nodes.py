# Define the state schema
import os
import re
import json
import io
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from matplotlib import pyplot as plt
from ultralytics import YOLO
from langchain_anthropic import ChatAnthropic
import boto3

from weather import WeatherData

from compute import full_image_processing_pipeline, read_image_from_s3, estimate_cwr_from_ndvi_and_weather, ndvi_to_kc


# S3 helper function for saving files
def save_to_s3(data, bucket_name, key, content_type=None):
    """
    Save data to an S3 bucket
    
    Parameters:
        data: The data to save (bytes or file-like object)
        bucket_name: Name of the S3 bucket
        key: S3 object key (path)
        content_type: MIME type of the data (optional)
    
    Returns:
        str: Full S3 URI of the saved object
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
        region_name=os.environ.get('AWS_REGION')
    )
    
    extra_args = {}
    if content_type:
        extra_args['ContentType'] = content_type
    
    s3_client.put_object(Body=data, Bucket=bucket_name, Key=key, **extra_args)
    return f"s3://{bucket_name}/{key}"


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
    data_summary: Dict[str, Any] = Field(default_factory=dict)  # Changed to accept a dict


class OutputStateModel(BaseModel):
    user_id: Optional[str] = None
    yolo_result: Optional[YoloAnalysisOutputModel] = None
    weather_data: Optional[WeatherDataOutputModel] = None
    ndvi_result: Optional[NDVIOutputModel] = None
    recommendation: Optional[MauiRecommendationModel] = None


# — define the one "State" model that has *all* fields —
class State(InputStateModel, OutputStateModel):
    pass


def check_for_tiff(input_state: InputStateModel):
    if input_state.image_key.endswith(".tiff"):
        return "NDVI_pipeline"
    else:
        return "YOLO_analysis"


def get_today_eto(state: State) -> float:
    try:
        today_weather_str = state.weather_data.past_7_days[-1]  # e.g., "2025-04-24: Temp ..., ETo 3.3mm"
        eto_match = re.search(r'ETo\s([\d.]+)mm', today_weather_str)
        if eto_match:
            return float(eto_match.group(1))
    except Exception as e:
        print(f"Error extracting today's ETo: {e}")
    return 0.0  # fallback


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
    output_bucket = "qijaniproductsbucket"  # Update with your output bucket name

    try:
        model = YOLO("yolo11s-pest-detection/best.pt")
        # For cloud deployment, we need to disable local saving
        results = model.predict(image, save=False)

        if not results:
            yolo_state = {"status": "No detections"}
            return yolo_state

        first_result = results[0]
        
        # Instead of saving locally, we'll render the annotated image and save to S3
        # Get the annotated image from the result
        annotated_img = first_result.plot()
        
        # Save the annotated image to S3
        img_bytes = io.BytesIO()
        plt.imsave(img_bytes, annotated_img, format='PNG')
        img_bytes.seek(0)
        
        # Generate S3 path
        base_filename = os.path.basename(image_path)
        s3_save_path = f"outputs/yolo_{base_filename}"
        
        # Save to S3
        s3_uri = save_to_s3(img_bytes, output_bucket, s3_save_path, content_type='image/png')

        yolo_state = {
            "status": "Success",
            "class_labels": first_result.names.values(),
            "scores": [float(c) for c in first_result.boxes.conf.tolist()],
            "bounding_boxes": [b.tolist() for b in first_result.boxes.xyxy],
            "save_path": s3_uri,
        }
        yolo_output = YoloAnalysisOutputModel(
            status=yolo_state["status"],
            class_labels=list(yolo_state["class_labels"]),  # convert dict_values to list
            scores=yolo_state["scores"],
            bounding_boxes=yolo_state["bounding_boxes"],
            save_path=yolo_state["save_path"]
        )

    except Exception as e:
        error_output = YoloAnalysisOutputModel(status=f"Error: {e}")
        return {"user_id": input_state.user_id, 
                "yolo_result": error_output.model_dump()}


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
    output_bucket = "qijaniproductsbucket"  # Update with your output bucket name

    try:
        image = read_image_from_s3("qijaniproductsbucket", input_state.image_key)
        ndvi_noise_reduced, _, kc_map = full_image_processing_pipeline(
            image,
            RADIOMETRIC_PARAMS,
            detector_type='ORB',
            noise_method='median',
            noise_kernel_size=3,
            sigma=1.0,
            nir_band_index=4,
            red_band_index=2,
            visualize=False,
            use_parallel_noise_reduction=False
        )

        # Generate base filename for S3
        base_filename = os.path.basename(input_state.image_key)
        s3_image_path = f"outputs/ndvi_{base_filename}.jpg"
        s3_npy_path = f"outputs/ndvi_{base_filename}.npy"

        # Save NDVI plot to S3
        plt.figure(figsize=(10, 8))
        plt.imshow(ndvi_noise_reduced, cmap='RdYlGn')
        plt.colorbar(label='NDVI Value')
        plt.title("NDVI Analysis")
        
        # Save plot to bytes buffer instead of file
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='jpg', dpi=300)
        img_buf.seek(0)
        plt.close()
        
        # Save image to S3
        s3_image_uri = save_to_s3(img_buf, output_bucket, s3_image_path, content_type='image/jpeg')
        
        # Save numpy array to S3
        npy_buf = io.BytesIO()
        np.save(npy_buf, ndvi_noise_reduced)
        npy_buf.seek(0)
        s3_npy_uri = save_to_s3(npy_buf, output_bucket, s3_npy_path, content_type='application/octet-stream')

        cwr = estimate_cwr_from_ndvi_and_weather(ndvi_to_kc(ndvi_noise_reduced), get_today_eto(input_state))
        ndvi_summary = {
            "min": float(np.min(ndvi_noise_reduced)),
            "max": float(np.max(ndvi_noise_reduced)),
            "mean": float(np.mean(ndvi_noise_reduced)),
            "cwr": cwr
        }

        return {
            "user_id": input_state.user_id,
            "ndvi_result": NDVIOutputModel(ndvi_summary=ndvi_summary, save_path=s3_npy_uri).model_dump()
        }

    except Exception as e:
        return {
            "user_id": input_state.user_id,
            "ndvi_result": {"error": str(e)}
        }


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

    # Fix the context string - use double curly braces to escape them in f-strings
    context = f"""
    You are an advanced agricultural intelligence agent built for precision farming.

    Analyze the provided field data — including drone-captured NDVI, pest detection, and live weather info (like rainfall and evapotranspiration). Think like a seasoned agronomist: connect the dots between crop health, pest patterns, and environmental factors to find actionable insights.

    When sharing findings with farmers, speak in a clear, friendly, no-jargon way. The farmer doesn't understand and cannot aggregate this data by themself and therefore avoid telling them about what the data shows, just what they need to do. Focus on helpful, practical advice they can apply immediately — whether for pest control, irrigation, or crop care.

    Clearly describe any pests detected (using their common English names), or likely threats based on the crop, weather, and location (latitude & longitude). Recommend appropriate treatments or pesticides farmers can safely use.

    Be specific. Name the pests likely or detected, and clearly suggest matching treatments (organic or chemical) by name.

    Your mission: empower farmers with clear, smart decisions they can act on now. The keyword? Actionable insights.
    
    Data provided:
    * Pest Detection: {', '.join([f"{label} ({score:.2f})" for label, score in zip(yolo_detection.class_labels, yolo_detection.scores)]) if yolo_detection else 'No pests detected'}
    * NDVI Summary: {ndvi_summary}
    * NDVI Image Path: {ndvi_image}
    * Weather Summary (Current): {weather_summary}
    
    Your output must be in the following JSON format:
    {{
        "risk_level": "Low|Moderate|High",
        "advice": "Actionable recommendation here",
        "data_summary": "Recap all findings in no more than 60 words"
    }}

    Make sure to follow this exact format, with these keys, and provide only valid JSON in your response.
    """
    
    # Use standard messages API instead of structured output
    response = model.invoke([{"role": "user", "content": context}])
    
    # Extract JSON response from the model output
    try:
        # Extract JSON content from response
        response_text = response.content
        # Find JSON content
        import json
        import re
        
        # Try to find JSON pattern in the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find a JSON object directly
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
                
        recommendation_data = json.loads(json_str)
        
        # Accommodate the string data_summary by converting it to a dict if necessary
        data_summary = recommendation_data.get("data_summary", "")
        if isinstance(data_summary, str):
            data_summary = {"summary": data_summary}
            
        recommendation = MauiRecommendationModel(
            risk_level=recommendation_data.get("risk_level", "Unknown"),
            advice=recommendation_data.get("advice", "No advice available"),
            data_summary=data_summary
        )
    except Exception as e:
        # Fallback if JSON parsing fails
        recommendation = MauiRecommendationModel(
            risk_level="Unknown",
            advice=f"Error parsing response: {str(e)}",
            data_summary={"error": "Failed to parse model output"}
        )
    
    return {"user_id": input_state.user_id, "recommendation": recommendation}