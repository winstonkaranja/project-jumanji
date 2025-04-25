# 🌾 Maui Agronomics Recommendation System (MARS)

MARS is a LangGraph-based agronomic assistant designed to help farmers make smarter decisions using imagery and weather data. It combines NDVI-based crop recommendations, pest detection powered by YOLOv11s (Small), and weather-aware insights (including agricultural evapotranspiration) into a single intelligent pipeline. The system uses Anthropic's Claude via LangChain to aggregate multimodal insights and output practical recommendations.

## 🧠 Key Features

- 🔍 **Pest Detection**: YOLOv11s model supports 102 pest classes. Supports all formats but pipeline only accepts jpeg and tiff/tif for now
- 🌱 **NDVI Pipeline**: Supports TIFF imagery, Calculates NDVI, CWR, and incorporates FAO Penman-Monteith evapotranspiration (ET₀) for precise irrigation insights.
- ☁️ **Weather-Aware Recommendations**: Factors in FAO Penman-Monteith evapotranspiration (ET₀) from open-meteo (https://open-meteo.com/en/docs)
- 🤖 **Maui Aggregation Agent**: Powered by ChatAnthropic via LangChain.
- ☁️ **Cloud Integration**: AWS S3 support for image storage and retrieval.

---

## 🛠 Tech Stack

- **Languages**: Python
- **Frameworks**: [LangChain](https://www.langchain.com/), [LangGraph](https://docs.langgraph.dev/), [LangSmith](https://smith.langchain.com/)
- **Vision Model**: YOLOv11s via [Ultralytics](https://github.com/ultralytics/ultralytics)

## Core Libraries
pydantic
requests
numpy
matplotlib
ultralytics
langchain-anthropic
langgraph
boto3
scikit-image
scipy
tifffile
Pillow
opencv-python


## ⚙️ Installation
Clone the repo

git clone https://github.com/your-username/mars.git
cd mars


Set up virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


Install dependencies

pip install -r requirements.txt
Configure environment variables

Create a .env file at the root with the following:

OPENAI_API_KEY=your_openai_key
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_langchain_project_name
LANGSMITH_TRACING=true
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
ANTHROPIC_API_KEY=your_anthropic_key


## 🚀 Usage

This project is designed to run as a LangGraph deployed pipeline, accessible via the LangChain SDK.

Query Maui with image inputs for pest detection (JPEG only) or NDVI-based recommendations + pest detection (TIFF).

Example

```python

from langgraph_sdk import Client

#Initialize the client
advisor_graph = Client(url = url, api_key = api_key)

#Simple invocation example
result = advisor_graph.invoke(
    {
        user_id="xyz",
        coordinates=LocationModel(latitude=37.7749, longitude=-122.4194),
        image_key="Maize.jpg"
    }
)

# Check what's actually in the result
print("Result structure:", result)

Result structure: {'user_id': 'xyz', 'yolo_result': {'status': 'Success', 'class_labels': ['rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer', 'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper', 'rice water weevil', 'rice leafhopper', 'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket', 'wireworm', 'white margined moth', 'black cutworm', 'large cutworm', 'yellow cutworm', 'red spider', 'corn borer', 'army worm', 'aphids', 'Potosiabre vitarsis', 'peach borer', 'english grain aphid', 'green bug', 'bird cherry-oataphid', 'wheat blossom midge', 'penthaleus major', 'longlegged spider mite', 'wheat phloeothrips', 'wheat sawfly', 'cerodonta denticornis', 'beet fly', 'flea beetle', 'cabbage army worm', 'beet army worm', 'Beet spot flies', 'meadow moth', 'beet weevil', 'sericaorient alismots chulsky', 'alfalfa weevil', 'flax budworm', 'alfalfa plant bug', 'tarnished plant bug', 'Locustoidea', 'lytta polita', 'legume blister beetle', 'blister beetle', 'therioaphis maculata Buckton', 'odontothrips loti', 'Thrips', 'alfalfa seed chalcid', 'Pieris canidia', 'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 'Colomerus vitis', 'Brevipoalpus lewisi McGregor', 'oides decempunctata', 'Polyphagotars onemus latus', 'Pseudococcus comstocki Kuwana', 'parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus', 'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 'Panonchus citri McGregor', 'Phyllocoptes oleiverus ashmead', 'Icerya purchasi Maskell', 'Unaspis yanonensis', 'Ceroplastes rubens', 'Chrysomphalus aonidum', 'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 'Tetradacus c Bactrocera minax', 'Dacus dorsalis(Hendel)', 'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii', 'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar', 'Salurnis marginella Guerr', 'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 'Sternochetus frigidus', 'Cicadellidae'], 'scores': [], 'bounding_boxes': [], 'save_path': 's3://qijaniproductsbucket/outputs/yolo_Maize.jpg'}, 'weather_data': {'location': 'Lat: 37.7749, Lon: -122.4194', 'past_7_days': ['2025-04-18: Temp 9.8-18.2°C, Humidity 82%, Rain 0.0mm, Wind Max 21.6 km/h, ETo 2.61mm', '2025-04-19: Temp 8.7-15.4°C, Humidity 85%, Rain 0.0mm, Wind Max 20.2 km/h, ETo 2.26mm', '2025-04-20: Temp 9.5-18.3°C, Humidity 83%, Rain 0.0mm, Wind Max 19.0 km/h, ETo 2.98mm', '2025-04-21: Temp 7.9-21.6°C, Humidity 78%, Rain 0.0mm, Wind Max 24.1 km/h, ETo 4.11mm', '2025-04-22: Temp 9.2-17.6°C, Humidity 83%, Rain 0.0mm, Wind Max 25.3 km/h, ETo 3.07mm', '2025-04-23: Temp 9.5-14.5°C, Humidity 81%, Rain 0.0mm, Wind Max 20.5 km/h, ETo 2.04mm', '2025-04-24: Temp 9.7-13.5°C, Humidity 79%, Rain 0.0mm, Wind Max 19.6 km/h, ETo 1.51mm'], 'current': {'temp': '9.2°C', 'wind': '13.5 km/h'}, 'forecast_next_days': ['2025-04-25: Temp 9.2-14.5°C, Humidity 75%, Rain 0.2mm, Wind Max 19.2 km/h, ETo 2.45mm', '2025-04-26: Temp 6.9-14.9°C, Humidity 76%, Rain 0.4mm, Wind Max 24.0 km/h, ETo 3.26mm', '2025-04-27: Temp 10.7-12.9°C, Humidity 77%, Rain 0.1mm, Wind Max 26.3 km/h, ETo 1.91mm']}, 'recommendation': MauiRecommendationModel(risk_level='Moderate', advice='Monitor your crops closely for signs of pest infestations. Consider applying an organic insecticide like neem oil or a targeted chemical pesticide to control any outbreaks. Maintain proper irrigation and soil moisture levels to support plant health.', data_summary={'summary': 'The weather data indicates moderate risk conditions, with potential for pest issues. No NDVI data is available, so crop health is unclear. Proactive monitoring and targeted pest management are recommended.'}), 'image_key': 'Maize.jpg', 'coordinates': LocationModel(latitude=37.7749, longitude=-122.4194)}

```


## 📁 Project Structure 

mars/
├── mars/                    # LangChain agents and Maui graph
├── compute.py               # NDVI, evapotranspiration, and CWR calculations and more!
├── nodes.py                 # Langchain node architecture
├── weather.py               # Helpers for weather info
├── .env                     # API and secret keys
├── .requirements.txt        # required packages to run the graph
├── recommendation.py        # Entry point (LangGraph workflow)
└── yolo11s-pest-detection/  # YOLO model integration

## 🪪 License
This project is licensed under the MIT License.


## 🤝 Contributing
Pull requests and issue reports are welcome. If you're working on precision agriculture, satellite imagery, or rural tech—let's collaborate.