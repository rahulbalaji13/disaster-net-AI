# Swarm-Enabled Edge Intelligence for Disaster Response IoT Systems

<img width="800" height="400" alt="Gemini_Generated_Image_cywpczcywpczcywp" src="https://github.com/user-attachments/assets/c2f3e7d3-84aa-4a0f-9973-4c9b25dfd8c3" />

Main idea: IoT device used to sense the type of disaster with environmental conditions and attached camera to capture surrounding and map with existing one.

Dataset Link ===> https://www.kaggle.com/datasets/rahul13289/disaster-swarm-edge-simulated-datasaet/data

Workflow:

<img width="800" height="400" alt="proj" src="https://github.com/user-attachments/assets/21d4ee06-219d-4421-825e-c70ca9ddd0ba" />


What is the possible type of disaster?
1.	Forest fire 
2.	Flood
3.	Earthquake 

Device Type	Purpose:

Environmental sensors	Fire, gas, humidity, temperature monitoring

Drones / Robots	Surveillance, navigation, payload delivery

Wearables	First responder health monitoring

Edge Gateways	Data aggregation and local AI processing

GPS Modules	Location tracking for rescue coordination

Disaster Type	Detection Devices / Methods	AI Training Capabilities	Response Actions (Drones / Robots / Edge):

🌋 Earthquake	Accelerometers (MPU6050), vibration sensors	Classify tremors, predict aftershocks	Swarm drones map damage; robots locate and rescue trapped individuals

🔥 Fire / Wildfire	Flame sensors, temperature sensors (DHT22), gas sensors (MQ2), cameras	Smoke/flame detection from vision data; temperature anomaly detection	Drones monitor fire spread with thermal imaging; robots aid in evacuation and safe pathfinding

🌊 Flood	Ultrasonic sensors (for water level), rainfall sensors, satellite + historical data	Predict flooding zones using edge-trained historical and real-time sensor data	Drones scout flooded zones; robots assist in shallow water rescue

💨 Cyclone / Hurricane	Barometric pressure sensors, weather APIs	Predict cyclone landfall zones and intensity	Drones monitor infrastructure; edge nodes send evacuation alerts

🌪️ Tornado	Wind sensors, pressure drop indicators, radar/satellite imaging	Train with wind pattern data and visual funnel cloud detection	Swarm bots assess debris field; alert system reroutes responders safely

🌧️ Landslide	Soil moisture + tilt sensors, vibration sensors	Slope failure prediction using ML based on weather + terrain	Drones assess slope safety; robots navigate unstable terrain

🔌 Industrial Accident	MQ2/MQ135 gas sensors, visual detection of leaks or smoke	Gas threshold anomaly detection and visual leak recognition	Robots contain spill or guide rescue; drones identify chemical zones

💥 Explosion / Bombing	Sound sensors, temperature spikes, sudden motion (shock sensors), visual analysis	AI trained on explosion patterns from camera feeds and sensor fusion	Drones provide overhead imaging; robots investigate debris zones

🏚️ Building Collapse	Accelerometers, vibration/load sensors, cameras	AI detects visual cracks, sudden structural movements	Robots locate survivors under rubble; drones guide emergency responders

☢️ Nuclear/Radiation Leak	Geiger counter, radiation sensor modules	Train AI models on radiation intensity, hotspot detection using distributed sensors	Robots enter radiation zones; drones measure air contamination

SAMPLE DATASET

•	Resilience under Constraint
Disasters often disrupt connectivity and limit power/resources—making edge-based, swarm-coordinated intelligence essential for sustained situational awareness and response.

•	Accurate, Real-Time Mapping & Detection
Visual change detection, coupled with environmental anomaly analysis, supports fast and reliable disaster classification and local impact assessment.

•	Adaptive, Collaborative Intelligence
Leveraging swarm and distributed learning offers robustness, scalability, and privacy-aware adaptation—even under heterogeneous and dynamic field conditions.

# Implementation

This repository includes a complete working implementation of the swarm-enabled edge intelligence system. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for details.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

## Features

- ✅ Disaster classification using sensor data (temperature, gas, humidity, flood levels)
- ✅ Edge computing nodes for distributed AI processing
- ✅ Swarm coordination with consensus-based decision making
- ✅ Support for multiple disaster types (fire, flood, earthquake, building collapse, etc.)
- ✅ Lightweight models suitable for IoT deployment
- ✅ Comprehensive unit tests (25 tests)

# Owner

Made with ❤️ by Rahul Balaji
