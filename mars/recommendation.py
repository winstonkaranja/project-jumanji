from langgraph.graph import StateGraph, START, END

from nodes import State, NDVI_analysis, YOLO_analysis, weather_node, check_for_tiff, Maui

# Create the state graph
graph = StateGraph(State)

# Add nodes
graph.add_node("NDVI_pipeline", NDVI_analysis)
graph.add_node("YOLO_analysis", YOLO_analysis)
graph.add_node("weather_node", weather_node)
graph.add_node("Maui", Maui)

# Define edges and conditional branching
graph.add_edge(START, "weather_node")
graph.add_conditional_edges("weather_node", check_for_tiff, ["NDVI_pipeline", "YOLO_analysis"])  # Checks for TIFF and routes accordingly
graph.add_edge("NDVI_pipeline", "YOLO_analysis")
graph.add_edge("YOLO_analysis", "Maui")
graph.add_edge("Maui", END)

# Compile the graph
advisor_graph = graph.compile()