# Smart Traffic Accident Prediction: Machine Learning Project

University Machine Learning project. The goal is to predict whether a **traffic accident was reported** (`accident_reported` = 0 / 1) from traffic, weather, and signal readings. It is a **binary classification** problem.

## Dataset

**Smart Traffic Management Dataset** (Kaggle): 2,000 one-minute readings from 5 locations, covering about 33 hours.

| Column | Description |
|---|---|
| `timestamp` | Reading time |
| `location_id` | Location code (1–5) |
| `traffic_volume` | Traffic volume |
| `avg_vehicle_speed` | Average speed (assumed km/h) |
| `vehicle_count_cars` / `_trucks` / `_bikes` | Vehicle counts by type |
| `weather_condition` | Sunny, Cloudy, Rainy, Foggy, Windy |
| `temperature`, `humidity` | Weather readings |
| `signal_status` | Green, Yellow, Red |
| `accident_reported` | **Target**: 1 = accident, 0 = no accident |

## Repository contents

```
├── Task1_Data_Preparation_EDA.ipynb   # Task 1 notebook: data understanding, cleaning, outliers, EDA
├── Task1_Code_Explanation.pdf         # Line-by-line explanation of every code cell (Arabic)
├── Dataset/
│   ├── smart_traffic_management_dataset.csv   # Raw data
│   └── traffic_cleaned_task1.csv              # Cleaned output of Task 1 (2000 × 16)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
```

Open `Task1_Data_Preparation_EDA.ipynb` in VS Code or Jupyter and choose **Run All**.

## Task 1: key findings

- **No** missing values, duplicate rows, or outliers. The outlier checks were IQR, Z-score, Isolation Forest, LOF, and domain checks.
- `timestamp` was converted from text to datetime. Three features were added: `hour`, `is_peak_hour`, and `total_vehicles`.
- The data is **highly imbalanced**: only 5.4% of rows are accidents (108 of 2000). Models will be evaluated with precision, recall, and F1, using a stratified split.
- `signal_status` is the only feature significantly associated with accidents (chi-square p = 0.038).
- The numeric features are uniformly distributed and barely correlated, so the data looks synthetic. Expect modest model performance.

## Next tasks

Encoding and scaling, then KNN, Decision Tree, and Naive Bayes models.
