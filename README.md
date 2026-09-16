# 🌱 Plant Survival AI

### AI-Based Plantation Recommendation System

Plant Survival AI is an AI-assisted plantation recommendation system that helps identify and rank tree species for a given location based on historical survival evidence, regional species occurrence, soil information, environmental factors, pollution-related evidence, and user-defined planting conditions.

The system is designed to answer:

> **"Which tree species are suitable candidates for planting at this location, and which ones have stronger evidence of survival and suitability?"**

---

## 🚀 Project Overview

Plant Survival AI is not a single machine-learning model. It is a multi-stage recommendation pipeline consisting of:

- 🌳 Historical tree survival prediction
- 📍 Regional species candidate selection
- 🌱 Soil image classification
- 🌦️ Climate/environmental information
- 🌫️ Pollution-related evidence
- 👤 User and site requirements
- 📊 Final species ranking

The main survival model is **V5**, based on **CatBoostRegressor**.

The recommendation and regional candidate-selection layer is **V7**.

---

## 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                     User Location
                           │
                           ▼
                Regional Species Data
                    ┌──────┴──────┐
                    │             │
                   GBIF      Local Census
                    │             │
                    └──────┬──────┘
                           ▼
                 Regional Species Pool
                           │
                           ▼
                 Taxonomy Normalization
                           │
                           ▼
                Survival Evidence Match
                    ┌──────┴──────┐
                    │             │
                Supported     Unsupported
                    │             │
                    ▼             ▼
                   V5        Regional-only
                    │
                    └──────┬──────┘
                           ▼
              Soil / Climate / Pollution
                           +
                      User Inputs
                           │
                           ▼
                 Recommendation Engine
                           │
                           ▼
                     Final Ranking
````

---

# 🧠 Main Machine Learning Model

## V5 Survival Model

The main survival model was trained using the historical dataset:

```text
df_survival_nov2022(1).csv
```

### Original dataset

| Property             |          Value |
| -------------------- | -------------: |
| Observations         |          5,294 |
| Columns              |             42 |
| Species              |            524 |
| Geographic locations |            112 |
| Target               | `survival_per` |

The dataset contains observations from South and Southeast Asia, including India, Malaysia, Indonesia, Thailand, China, Sri Lanka, the Philippines, Laos and Vietnam.

---

## Data preprocessing

Observations with:

```text
duration_months = 0
```

were treated as baseline observations and removed.

```text
Original observations       5,294
Baseline observations       1,917
Post-baseline observations  3,377
```

The final modelling dataset therefore contains **3,377 observations**.

---

## Leakage prevention

The following variables were excluded:

```text
number_alive
number_dead
number_monitored
```

These variables can contain information closely related to the observed survival outcome and could introduce target leakage.

---

## V5 Features

The V5 model uses 28 features:

```text
country
province
disturbance
forest_condition
comp_removal
shading
soil_prep
water_reg
fertilisation
protection
planting_density
planting_sp_no
age_0
height_0
species_full
genus
species
family
lat_dec
lon_dec
duration_months
w_meanWD
w_sdWD
w_nInd
treatment
lat_squared
lon_squared
lat_lon_interaction
```

---

## Model Configuration

**Algorithm:** CatBoostRegressor

```text
Iterations     : 1200
Depth          : 7
Learning Rate  : 0.035
Loss Function  : RMSE
```

The trained model is stored as:

```text
models/survival_v5_CORRECT.cbm
```

---

## 📊 V5 Model Performance

The model was evaluated using geographic holdout validation.

| Metric |                   Result |
| ------ | -----------------------: |
| MAE    | 18.033 percentage points |
| RMSE   | 22.514 percentage points |
| R²     |                    0.060 |

Geographic validation was used instead of a simple random row split to provide a more realistic evaluation on unseen geographic locations.

### Important interpretation

The V5 output should be treated as a:

> **Survival Suitability Score**

and **not** as a calibrated probability of survival.

For example, a score of 89 does not mean that the tree has an 89% guaranteed probability of surviving in a particular city.

---

# 📍 Regional Candidate Selection

A survival model cannot recommend every tree species in the world because the training dataset contains a limited number of species and locations.

Therefore, the system first needs to determine which species are relevant to the user's region.

## Pune prototype

The Pune Tree Census contains approximately:

* **4 million tree records**
* **418 normalized species**
* **17 CSV files**

The Pune species were matched with the historical survival dataset.

This produced a local candidate pool of:

**16 species**

The current V7 Pune implementation identifies six species as:

```text
A_ML_SUPPORTED
```

These have both regional evidence and matching historical survival evidence.

---

# 🌍 GBIF Integration

The project also investigates **GBIF** as an open biodiversity data source for regional species discovery.

The intended workflow is:

```text
User Location
      ↓
GBIF Regional Occurrence Data
      ↓
Species Recorded in Region
      ↓
Regional Candidate Pool
      ↓
Match With Survival Evidence
      ↓
V5 / Evidence-Based Ranking
```

GBIF provides **species occurrence evidence**.

It does not directly provide:

* Survival probability
* Urban planting suitability
* Water requirements
* Pollution removal capability
* Maintenance requirements

Therefore, GBIF is treated as a **regional candidate/evidence source**, not as a complete suitability model.

---

# 🌱 Soil Image AI

Plant Survival AI also contains a separate soil classification model.

### Dataset

The soil model was trained using an Indian soil image dataset containing:

```text
1,189 images
7 soil classes
```

Classes:

```text
Alluvial
Arid
Black
Laterite
Mountain
Red
Yellow
```

### Model

```text
ImageNet-pretrained EfficientNet-B0
```

### Test performance

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 84.92% |
| Balanced Accuracy | 86.11% |
| Macro F1          | 82.56% |

The model predicts a soil class and confidence score.

Example:

```text
Soil Image
    ↓
EfficientNet-B0
    ↓
Black Soil
87.5% confidence
```

The soil model is **not** intended to directly predict laboratory measurements such as exact pH, nitrogen, phosphorus or potassium from an RGB image.

---

# 🌦️ Climate Layer

Climate information has also been investigated as an additional suitability factor.

The project tested variables including:

```text
Temperature
Maximum Temperature
Minimum Temperature
Precipitation
Relative Humidity
Solar Radiation
Wind Speed
Evapotranspiration
```

NASA POWER was used during the environmental experiment.

However, adding annual climate variables directly to V5 reduced test performance.

Therefore, future versions should investigate:

* Seasonal climate
* Dry spells
* Heat stress
* Seasonal rainfall
* Water stress
* Extreme temperature
* Evapotranspiration

rather than relying only on annual averages.

---

# 🌫️ Pollution Layer

Pollution can influence the ranking of species.

Potential pollutants include:

```text
PM2.5
PM10
NO2
SO2
O3
```

The conceptual approach is:

```text
Pollution severity
        ×
Species pollution capability
        ↓
Pollution contribution
```

However, pollution removal, particulate interception and pollution tolerance are different concepts.

Therefore, the project does **not** claim unsupported universal values such as:

> "Species A absorbs 5× more pollution than Species B."

Future versions will use pollutant-specific scientific evidence and preserve the measurement units and study conditions.

---

# 👤 User Inputs

The recommendation system is designed to consider practical planting requirements such as:

### Water availability

```text
Low
Medium
High
```

### Sunlight

```text
Low
Medium
High
```

### Maintenance

```text
Low
Medium
High
```

### Site type

Examples:

```text
Roadside
Residential
Park
Industrial
School / Campus
Agricultural / Open Land
```

### Planting purpose

Examples can include:

```text
Pollution reduction
Shade
Biodiversity
Aesthetic planting
Roadside plantation
General plantation
```

These inputs allow the system to consider the actual planting context rather than relying only on historical survival data.

---

# 🔄 Complete Recommendation Pipeline

The current concept is:

```text
                    USER
                     │
                     ▼
                Location
                     │
                     ▼
          Regional Species Data
             ┌───────┴───────┐
             │               │
            GBIF       Local Census
             │               │
             └───────┬───────┘
                     ▼
            Regional Species Pool
                     │
                     ▼
            Taxonomy Normalization
                     │
                     ▼
           Survival Evidence Match
                     │
              ┌──────┴──────┐
              │             │
          Supported     Unsupported
              │             │
              ▼             ▼
             V5        Regional-only
              │
              └──────┬──────┘
                     ▼
        Soil / Climate / Pollution
                     +
                User Inputs
                     │
                     ▼
            Recommendation Engine
                     │
                     ▼
                Final Ranking
```

---

# 🗂️ Project Structure

```text
Plant_Survival_Ai/
│
├── data/
│   ├── pollution_capability.csv
│   ├── pune_final_candidate_pool.csv
│   ├── survival_ml_cleaned.csv
│   ├── survival_train.csv
│   ├── survival_validation.csv
│   └── survival_test.csv
│
├── models/
│   ├── survival_v5_CORRECT.cbm
│   ├── survival_ranking_v1.cbm
│   └── soil/
│       ├── soil_efficientnet_b0_final.pt
│       └── soil_model_metrics.json
│
├── outputs/
│   ├── v5_test_predictions_for_ranking.csv
│   ├── v5_ranking_evaluation.csv
│   ├── v6_final_ranking.csv
│   ├── v7_final_ranking.csv
│   ├── v7_gbif_bounded_records.csv
│   ├── v7_gbif_discovered_species.csv
│   ├── v7_main_recommendation_pool.csv
│   └── v7_regional_only_candidates.csv
│
├── src/
│   ├── predict_candidates.py
│   ├── predict_soil_multi.py
│   ├── soil_predictor.py
│   ├── soil_compatibility.py
│   ├── species_matching.py
│   ├── evidence_scoring.py
│   ├── pollution.py
│   ├── v6_recommender.py
│   ├── v7_regional_candidates.py
│   └── v7_final_recommender.py
│
├── index.html
├── results.html
├── script.js
├── results.js
├── api.php
├── style.css
└── species.json
```

---

# 💻 Technology Stack

### Machine Learning

* Python
* CatBoost
* PyTorch
* TorchVision
* Scikit-learn
* Pandas
* NumPy

### Computer Vision

* EfficientNet-B0
* ImageNet pretrained weights
* PyTorch

### Data Sources

* Historical tree survival dataset
* Pune Tree Census
* GBIF regional occurrence data
* Soil image dataset
* NASA POWER environmental data
* Scientific literature for pollution-related evidence

### Web Application

* HTML
* CSS
* JavaScript
* PHP
* XAMPP

The current prototype intentionally uses a simple **HTML + CSS + JavaScript + PHP** architecture instead of a complex React/FastAPI stack.

---

# ⚠️ Current Limitations

The project is currently a research/prototype system and has several limitations.

### 1. Geographic generalization

The V5 model has limited performance on unseen geographic locations.

The current test result:

```text
R² = 0.060
```

indicates that significant variation in survival remains unexplained.

### 2. Pune dependency

The current V7 implementation is still partly dependent on the Pune candidate pool.

Therefore, the current implementation is **not completely location-independent**.

### 3. Limited survival data

The historical survival dataset does not contain sufficient survival evidence for every species and every geographic region.

### 4. GBIF limitation

GBIF occurrence data indicate recorded presence but do not prove planting suitability.

### 5. Soil AI limitation

The soil classifier requires external validation on real-world smartphone photographs from different locations, lighting conditions and cameras.

### 6. Soil chemistry limitation

A normal RGB photograph should not be treated as an accurate laboratory soil test.

### 7. Pollution limitation

Pollution-removal studies use different pollutants, experimental methods and environmental conditions, making direct comparison difficult.

### 8. Climate limitation

Annual climate averages may not represent seasonal conditions, drought or heat stress.

### 9. Score interpretation

The final recommendation score is a ranking/suitability score and is not a calibrated survival probability.

### 10. Taxonomy

Species names require continued normalization, including synonyms and taxonomic changes.

---

# 🔮 Future Improvements

## Dynamic regional candidate generation

The most important improvement is to remove the Pune-specific candidate bottleneck.

Instead of:

```text
Pune → Fixed 16 species
```

the future system should use:

```text
User Location
      ↓
Regional Biodiversity Data
      ↓
Regional Species Pool
```

This will allow the system to operate across different cities and regions.

---

## Expand survival datasets

Future training data should include:

* Indian plantation studies
* Urban plantation datasets
* Restoration projects
* Municipal planting programs
* Long-term survival monitoring
* Different climate zones
* Different soil conditions

---

## Improve spatial validation

Future models should be evaluated separately across:

* Countries
* Regions
* Climate zones
* Urban environments
* Rural environments
* Soil classes

This will provide a better understanding of where the model performs reliably.

---

## Improve environmental modelling

Future versions should incorporate:

* Monthly climate
* Seasonal rainfall
* Dry-spell duration
* Heat stress
* Water stress
* Extreme temperatures
* Seasonal evapotranspiration

---

## Improve soil AI

Future soil development should collect:

```text
Smartphone soil photograph
          +
GPS/location
          +
Laboratory soil measurements
```

This would enable development of a more robust soil intelligence system.

---

## Improve pollution modelling

The future pollution database should distinguish between:

```text
PM2.5 removal
PM10 removal
NO2 response
SO2 response
O3 response
Particulate interception
Pollution tolerance
```

Each measurement should retain its source, units, study conditions and evidence quality.

---

## Add practical planting constraints

Future recommendations can consider:

* Root behavior
* Mature canopy size
* Mature height
* Water requirements
* Maintenance
* Infrastructure compatibility
* Invasive status
* Local availability
* Biodiversity value

---

## Real-World Feedback Loop

A long-term goal is to collect actual plantation outcomes.

```text
Recommendation
      ↓
Actual Plantation
      ↓
Monitoring
      ↓
Observed Survival
      ↓
New Training Data
      ↓
Model Improvement
      ↓
Better Recommendations
```

This would allow the system to continuously improve using real-world evidence.

---

# 📌 Project Status

### Current prototype

| Component                                 | Status    |
| ----------------------------------------- | --------- |
| Historical survival preprocessing         | ✅         |
| V5 CatBoost survival model                | ✅         |
| Geographic validation                     | ✅         |
| Pune Tree Census integration              | ✅         |
| V7 regional candidate experiment          | ✅         |
| GBIF regional evidence experiment         | ✅         |
| Soil image classifier                     | ✅         |
| Soil multi-image prediction               | ✅         |
| Climate experiment                        | ✅         |
| Pollution layer prototype                 | ✅         |
| HTML/CSS/JS frontend                      | ✅         |
| PHP API                                   | ✅         |
| Location-independent candidate generation | 🚧 Future |
| Large-scale real-world validation         | 🚧 Future |

---

# 🎯 Current Objective

The immediate goal is to develop a working prototype that can:

1. Accept a user's location
2. Identify regionally relevant tree species
3. Match species with historical survival evidence
4. Generate survival suitability scores
5. Optionally analyze soil images
6. Consider environmental and user/site factors
7. Rank suitable tree species
8. Explain why each species was recommended

---

# 📖 Important Disclaimer

Plant Survival AI is a research and prototype project.

The recommendations should be treated as **decision-support information**, not as a guarantee that a particular tree will survive at a specific location.

The quality of a recommendation depends on the availability, quality and geographic relevance of the underlying data.

Professional ecological, horticultural or forestry assessment may still be required for actual large-scale plantation decisions.

---

# 🌱 Vision

The long-term vision of Plant Survival AI is:

> **A location-independent AI system that combines biodiversity data, historical survival evidence, soil intelligence, climate, pollution and practical planting constraints to provide transparent, evidence-based tree recommendations for different regions.**

```text
Location
   +
Environment
   +
Soil
   +
Pollution
   +
Planting Conditions
   +
Historical Evidence
        ↓
   Plant Survival AI
        ↓
Recommended Trees
        +
Suitability
        +
Evidence
        +
Uncertainty
```

---

## 👨‍💻 Project

**Plant Survival AI**
AI-Based Plantation Recommendation System

Built using Python, CatBoost, PyTorch, EfficientNet, JavaScript, PHP and open environmental/biodiversity data sources.

````

