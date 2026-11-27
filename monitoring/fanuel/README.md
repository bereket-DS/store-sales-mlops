# Monitoring & Observability Stack (Fanuel)

This directory contains the complete **Application Logging & Monitoring** implementation for the Store Sales forecasting model. It fulfills the observability requirements by tracking infrastructure health, API metrics, and Machine Learning-specific performance (latency and predictions).

---

## 📁 Folder Structure

*   **`api.py`**: The FastAPI application that serves the trained LightGBM `.pkl` model, instrumented with Prometheus metrics and Loki logging.
*   **`prometheus.yml`**: Configuration telling Prometheus where to scrape the API metrics.
*   **`datasources.yml`**: Grafana provisioning file to automatically connect Prometheus and Loki.
*   **`dashboards.yml`**: Grafana provisioning file to automatically load the dashboard JSON.
*   **`grafana_dashboard.json`**: The exported blueprint of the 5-panel observability dashboard.
*   **`grafana_dashboard.md`**: A detailed technical report describing each visualization and its importance.
*   **`drift_report.py`**: A script that generates offline data drift and prediction distribution analysis.
*   **`distribution_drift.png`**: The generated plot comparing training vs. validation distributions.
*   **`dashboard_images/`**: Folder containing screenshots of the live Grafana dashboard for the final report.

---

##  How to Run the Monitoring Stack

The entire stack (API, Prometheus, Grafana, Loki) is containerized using Docker. 

1. Ensure **Docker Desktop** is running.
2. From the root of the repository, run:
   ```bash
   docker compose -f docker/fanuel/docker-compose.yml up -d --build
3. Wait a minute for the containers to start:

---

##  🌐 Accessing the Services 
Once running, the services are available at:

* FastAPI Swagger UI (Test Predictions): http://localhost:8000/docs
* Prometheus (Raw Metrics): http://localhost:9090
* Grafana (Visualization): http://localhost:3000
   * Username: admin
   * Password: admin

Note: The Grafana data sources and dashboard are auto-provisioned. When you log in, simply navigate to
      
      Dashboards -> Fanuel -> Fanuel API Monitoring to see the live metrics.

---

##  🧠 ML-Specific Monitoring Features
This implementation includes ML-specific metrics:
* Inference Latency (model_prediction_latency_seconds): Tracks how fast the model returns a prediction.
* Prediction Distribution (model_prediction_value): Monitors the actual output of the model to detect Concept Drift or Data Drift over time.
* API Live Logs: Streamed in real-time using Grafana Loki, capturing inference requests and system events.
---

##  📝 Detailed Report
For a deep dive into each of the 5 visualizations (including screenshots and technical descriptions), please refer to the grafana_dashboard.md file.