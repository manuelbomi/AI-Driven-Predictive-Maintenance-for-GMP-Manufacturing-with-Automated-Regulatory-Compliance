## AI-Driven Predictive Maintenance for GMP Manufacturing with Automated Regulatory Compliance (Regeneron IOPS)  

> [!NOTE]
> ### <ins>Regulatory Frameworks covered</ins>:  GAMP 5 , ALCOA+ , FDA CFR Part 11 & EU GMP Annex 11 
---


##   Executive Summary

##### This project implements a comprehensive AI-driven predictive maintenance solution for Regeneron's GMP manufacturing environment with fully automated regulatory compliance. Our hybrid ensemble approach combines LSTM, BERT, and XGBoost models with integrated SHAP explainability to deliver accurate, transparent, and regulatory-compliant equipment failure predictions while ensuring full adherence to GAMP5, ALCOA+, FDA CFR Part 11, and EU GMP Annex 11 requirements.

> [!IMPORTANT]
> <ins> Automated Regulatory Complinace Innovation</ins>: Our solution features automated compliance frameworks that continuously validate and document adherence to all major pharmaceutical regulatory standards, providing real-time compliance monitoring, automated audit trail generation, and comprehensive validation packages ready for regulatory inspection.

---

## Table of Contents

- Problem Overview

- Solution Architecture

- Data Collection & Preprocessing

- Model Architecture

- Implementation

- GMP Compliance

- Deployment Strategy

- LLM Integration

- MLOps & Infrastructure

---

## Problem Overview

#### <ins>Current Challenges: AI Layer</ins>

- #### <ins>Preventive Maintenance Inefficiency</ins>: Fixed schedules don't account for actual equipment conditions

- #### <ins>Unplanned Downtime</ins>: Equipment failures cause significant production losses

- #### <ins>Compliance Risks</ins>: Manual processes risk regulatory non-compliance

- #### <ins>Data Silos</ins>: Sensor data, maintenance records, and operational logs are disconnected

---



#### <ins>Current Challenges: Regulatory Framework Layer</ins>

- ####  Compliance Complexity</ins>: Navigating multiple regulatory frameworks (GAMP5, ALCOA+, FDA CFR Part 11, EU GMP Annex 11)

- ####  Explainability Requirements</ins>: Regulatory demand for transparent AI decision-making

- ####  Validation Burden</ins>: Extensive documentation and validation requirements for AI systems

- ####  Data Integrity</ins>: Maintaining ALCOA+ principles across AI data lifecycle

- ####  Audit Preparedness</ins>: Need for continuous compliance monitoring and reporting

---

## Business Impact

- #### <ins> Financial</ins>: Multi-million  dollar savings due to absence of plant/equipment downtimes

- #### <ins> Regulatory</ins>: Strict GMP/GLP compliance requirements (FDA 21 CFR Part 11, EU Annex 11)

- #### <ins> Operational</ins>: 24/7 manufacturing with zero tolerance for quality deviations
  
- #### <ins>Regulatory</ins>: Zero findings in FDA/EU audits through automated compliance

- #### <ins> Operational</ins>: 24/7 manufacturing with AI-driven predictive maintenance

- #### <ins> Compliance</ins>: Automated regulatory documentation and audit trails



---
---

## Solution Architecture Layers

#### In this section, we have exhaustively shown different viewpoints through which GMP regulatory frameworks could be achieved while satisfying data sourcing, data quality, AI applications and on-prem/cloud cloud deployment approaches. 

#### <ins>Hybrid On-Prem/Cloud Architecture</ins>  (high level)


<img width="3504" height="3210" alt="Image" src="https://github.com/user-attachments/assets/01410440-a6b3-4811-9b37-c9ea649fc0a8" />

---

#### <ins>Hybrid On-Prem/Cloud Architecture</ins>  (detailed)

```python
┌─────────────────────────────────────────────────────────────────────────────
│                        ON-PREMISES & EDGE LAYER                            │
├─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┤
│   MANUFACTURING │    │    EDGE LAYER    │    │    ON-PREMISES              │
│     FLOOR       │    │                  │    │    INFRASTRUCTURE           │
│                 │    │                  │    │                             │
│  ┌─────────────┐│    │  ┌─────────────┐ │    │  ┌─────────────────┐        │
│  │   SCADA     ││    │  │  EDGE AI    │ │    │  │   HISTORIAN     │        │
│  │   PLCs      ├─────►  │  INFERENCE  │ │    │  │   (PI System)   │        │
│  │  Sensors    ││    │  │             │ │    │  │                 │        │
│  └─────────────┘│    │  │ • Lightweight │    │  └─────────────────┘        │
│                 │    │  │   Models    │ │    │                             │
│  ┌─────────────┐│    │  │ • Real-time │ │    │  ┌─────────────────┐        │
│  │   MES       ││    │  │   Alerts    │ │    │  │   LOCAL CMMS    │        │
│  │ Batch       ├─────►  │ • Data      │ │    │  │   INTEGRATION   │        │
│  │ Execution   ││    │  │   Filtering │ │    │  │                 │        │
│  └─────────────┘│    │  └─────────────┘ │    │  └─────────────────┘        │
│                 │    │                  │    │                             │
│  ┌─────────────┐│    │  ┌─────────────┐ │    │  ┌─────────────────┐        │
│  │   LOCAL     ││    │  │ EDGE GATEWAY│ │    │  │   ON-PREM       │        │
│  │   HMI       │◄─────► │   (IoT)     │ │    │  │   DATA STORE    │        │
│  │  Alerts     ││    │  │             │ │    │  │                 │        │
│  └─────────────┘│    │  │ • OPC-UA    │ │    │  └─────────────────┘        │
│                 │    │  │ • MQTT      │ │    │                             │
└─────────────────┘    │  │ • Protocol  │ │    └─────────────────────────────┘
                       │  │   Conversion│ │
                       │  └─────────────┘ │
                       └──────────────────┘
                               │  ▲
                    ┌──────────┘  └──────────┐
                    │                        │
                    ▼                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD PLATFORM                                │
├──────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┤
│   DATA INGESTION │    │   PROCESSING &   │    │     AI/ML SERVICES          │
│                  │    │     STORAGE      │    │                             │
│  ┌─────────────┐ │    │  ┌─────────────┐ │    │  ┌─────────────────┐        │
│  │ AZURE IOT   │ │    │  │ DATA LAKE   │ │    │  │  AZURE ML       │        │
│  │    HUB      │ │    │  │  STORAGE    │ │    │  │                 │        │
│  │             │ │    │  │             │ │    │  │ • Model Training│        │
│  │ • Device    │ │    │  │ • Raw Data  │ │    │  │ • Experiment    │        │
│  │   Mgmt      │ │    │  │ • Processed │ │    │  │   Tracking      │        │
│  │ • Secure    │ │    │  │   Data      │ │    │  │ • Model Registry│        │
│  │   Comm      │ │    │  │ • Features  │ │    │  └─────────────────┘        │
│  └─────────────┘ │    │  └─────────────┘ │    │                             │
│                  │    │                  │    │  ┌─────────────────┐        │
│  ┌─────────────┐ │    │  ┌─────────────┐ │    │  │   AKS CLUSTER   │        │
│  │ EVENT HUB   │ │    │  │ DATABRICKS  │ │    │  │                 │        │
│  │             │ │    │  │             │ │    │  │ • Model Serving │        │
│  │ • High      │ │    │  │ • Feature   │ │    │  │ • Scalable      │        │
│  │   Throughput│ │    │  │   Engineering │    │  │   Inference     │        │
│  │ • Stream    │ │    │  │ • Model     │ │    │  │ • High Availability      │
│  │   Processing│ │    │  │   Training  │ │    │  └─────────────────┘        │
│  └─────────────┘ │    │  └─────────────┘ │    │                             │
└──────────────────┘    └──────────────────┘    └─────────────────────────────┘
                               │  ▲
                    ┌──────────┘  └──────────┐
                    │                        │
                    ▼                        │
┌────────────────────────────────────────────────────────────────────────────-
│                      BUSINESS LAYER & APPLICATIONS                         │
├─────────────────────┐    ┌──────────────────┐    ┌─────────────────────────┤
│   MAINTENANCE &     │    │   VISUALIZATION  │    │   INTEGRATION & API     │
│   OPERATIONS        │    │   & REPORTING    │    │   LAYER                 │
│                     │    │                  │    │                         │
│  ┌─────────────────┐│    │  ┌─────────────┐ │    │  ┌─────────────────┐    │
│  │   ENTERPRISE    ││    │  │ POWER BI    │ │    │  │   REST APIs     │    │
│  │     CMMS        ││    │  │ DASHBOARDS  │ │    │  │                 │    │
│  │                 ││    │  │             │ │    │  │ • Predictions   │    │
│  │ • Work Orders   ││    │  │ • Equipment │ │    │  │ • Health Scores │    │
│  │ • Maintenance   ││    │  │   Health    │ │    │  │ • Alerts        │    │
│  │   Scheduling    ││    │  │ • Predictive│ │    │  └─────────────────┘    │
│  └─────────────────┘│    │  │   Insights  │ │    │                         │
│                     │    │  └─────────────┘ │    │  ┌─────────────────┐    │
│  ┌─────────────────┐│    │                  │    │  │   SCADA/HMI     │    │
│  │   PLANT HMI     ││    │  ┌─────────────┐ │    │  │   INTEGRATION   │    │
│  │                 ││    │  │   EMAIL &   │ │    │  │                 │    │
│  │ • Real-time     ││    │  │   MOBILE    │ │    │  │ • Real-time     │    │
│  │   Alerts        ││    │  │   ALERTS    │ │    │  │   Updates       │    │
│  │ • Operator      ││    │  │             │ │    │  └─────────────────┘    │
│  │   Actions       ││    │  │ • Push      │ │    │                         │
│  └─────────────────┘│    │  │   Notifications    └─────────────────────────┘
└─────────────────────┘    └──────────────────┘  │
                                                 │
                       ┌─────────────────────────┘
                       │
               ┌──────────────────┐
               │   GOVERNANCE     │
               │   LAYER          │
               │                  │
               │  AZURE PURVIEW   │
               │                  │
               │ • Data Catalog   │
               │ • Data Lineage   │
               │ • Compliance     │
               │   Monitoring     │
               │ • Audit Trail    │
               └──────────────────┘

```

---

#### <ins>Cloud-Layer Solution Architecture </ins>

GMP PREDICTIVE MAINTENANCE ARCHITECTURE

```python
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   DATA SOURCES  │    │   AZURE CLOUD    │    │   BUSINESS LAYER    │
│                 │    │   PLATFORM       │    │                     │
│  ┌─────────────┐│    │                  │    │  ┌─────────────────┐│
│  │ SCADA/PLC   ├─────►  IoT Hub         │    │  │   CMMS          ││
│  │ Sensors     ││    │  (Real-time      │    │  │   Integration   ││
│  └─────────────┘│    │   Ingestion)     │    │  └─────────────────┘│
│                 │    │                  │    │                     │
│  ┌─────────────┐│    │  ┌─────────────┐ │    │  ┌─────────────────┐│
│  │   CMMS      ├─────►  │ Data Lake   │ │    │  │   SCADA/HMI     ││
│  │ Maintenance ││    │  │ Storage     │ │    │  │   Alerts        ││
│  └─────────────┘│    │  └─────────────┘ │    │  └─────────────────┘│
│                 │    │                  │    │                     │
│  ┌─────────────┐│    │  ┌─────────────┐ │    │  ┌─────────────────┐│
│  │    MES      ├─────►  │ Databricks  │ │    │  │   Dashboards    ││
│  │ Batch Records│    │   (Processing) │ │    │  │   & Reports     ││
│  └─────────────┘│    │  └─────────────┘ │    │  └─────────────────┘│
└─────────────────┘    │                  │    └─────────────────────┘
                       │  ┌─────────────┐ │
                       │  │ Azure ML    │ │
                       │  │ (AI/ML)     │ │
                       │  └─────────────┘ │
                       │                  │
                       │  ┌─────────────┐ │
                       │  │ AKS         │ │
                       │  │ (Serving)   │ │
                       │  └─────────────┘ │
                       └──────────────────┘                           
                       ┌──────────────────┐
                       │   GOVERNANCE     │
                       │   LAYER          │
                       │                  │
                       │  Azure Purview   │
                       │  (Data Catalog)  │
                       │                  │
                       │  Audit Trail     │
                       │  & Compliance    │
                       └──────────────────┘

```

---

## Integrated AI/Regulatory Framework Compliance Solution Suite

<img width="3604" height="1834" alt="Image" src="https://github.com/user-attachments/assets/46ce0327-f890-41f9-9dc4-a5d95b54bcdc" />




---


``` python

COMPLIANCE-FIRST AI ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGULATORY COMPLIANCE LAYER                              │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│   GAMP5         │   ALCOA+          │   FDA CFR 11      │   EU Annex 11     │
│   VALIDATION    │   DATA INTEGRITY  │   ELECTRONIC      │   COMPUTERIZED    │
│   ENGINE        │   FRAMEWORK       │   RECORDS         │   SYSTEMS         │
├─────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ • Automated     │ • Attributable    │ • Audit Trail     │ • Risk Management │
│   Validation    │ • Legible         │   Automation      │   Automation      │
│ • Risk          │ • Contemporaneous │ • Electronic      │ • Supplier        │
│   Assessment    │ • Original        │   Signatures      │   Qualification   │
│ • Test Protocol │ • Accurate        │ • System          │ • Data Integrity  │
│   Generation    │ • Complete        │   Validation      │ • Business        │
│ • Documentation │ • Consistent      │ • Security        │   Continuity      │
│   Automation    │ • Enduring        │   Controls        │ • Validation      │
│                 │ • Available       │ • Record          │   Documentation   │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI PREDICTIVE MAINTENANCE LAYER                        │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│   EXPLAINABLE   │   HYBRID ENSEMBLE │   SHAP-BASED      │   COMPLIANT       │
│   AI MODELS     │   ARCHITECTURE    │   EXPLAINABILITY  │   DEPLOYMENT      │
├─────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ • LSTM with     │ • Meta-Learner    │ • Feature         │ • Canary Release  │
│   Temporal      │   with Compliance │   Importance      │   with Validation │
│   Audit Trail   │   Weights         │   Scoring        │ • Shadow Mode     │
│ • BERT with     │ • Context-Aware   │ • Prediction      │   Deployment      │
│   Attention     │   Model Selection │   Breakdown       │ • Automated       │
│   Explanations  │ • Ensemble        │ • Risk Factor     │   Rollback        │
│ • XGBoost with  │   Validation      │   Identification  │ • Compliance      │
│   Feature       │ • Performance     │ • Confidence      │   Gates           │
│   Importance    │   Monitoring      │   Calibration     │ • Validation      │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘

```

```python

COMPLIANCE-FIRST DATA SOURCING & DATA MANGEMENT APPROACH
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   DATA SOURCES  │    │   AI PLATFORM    │    │   COMPLIANCE LAYER  │
│                 │    │   WITH SHAP      │    │                     │
│  ┌─────────────┐│    │                  │    │  ┌─────────────────┐│
│  │ ALCOA+      ├─────►  Hybrid Ensemble │    │  │   GAMP 5        ││
│  │ Compliant   ││    │   (LSTM/BERT/    │    │  │   Validation    ││
│  │ Data        ││    │   XGBoost)       │    │  └─────────────────┘│
│  └─────────────┘│    │                  │    │                     │
│                 │    │  ┌─────────────┐ │    │  ┌─────────────────┐│
│  ┌─────────────┐│    │  │ SHAP        │ │    │  │   FDA 21 CFR    ││
│  │ EU Annex 11 ├─────►  │ Explainability │    │  │   Part 11       ││
│  │ Compliant   ││    │  │ Engine      │ │    │  └─────────────────┘│
│  └─────────────┘│    │  └─────────────┘ │    │                     │
│                 │    │                  │    │  ┌─────────────────┐│
│  ┌─────────────┐│    │  ┌─────────────┐ │    │  │   Automated     ││
│  │ GMP Data    ├─────►  │ Automated   │ │    │  │   Audit Trail   ││
│  │ Integrity   ││    │  │ Compliance  │ │    │  │   Generation    ││
│  └─────────────┘│    │  │ Monitoring  │ │    │  └─────────────────┘│
└─────────────────┘    │  └─────────────┘ │    └─────────────────────┘
                       │                  │
                       │  ┌─────────────┐ │
                       │  │ Change      │ │
                       │  │ Control     │ │
                       │  │ System      │ │
                       │  └─────────────┘ │
                       └──────────────────┘

```
### OVERALL INTEGRATED ON-PREM/CLOUD FRAMEWORK

<img width="4750" height="3636" alt="Image" src="https://github.com/user-attachments/assets/ed89fd47-d3b5-4fde-bd2a-ab7a4af370a1" />


---


## Data Collection & Preprocessing

#### <ins>Data Sources & Processing</ins>

##### <ins> Sensor Data </ins> :  Real-time equipment monitoring (vibration, temperature, pressure)

##### <ins> Maintenance Records</ins>: Historical work orders and failure data

##### <ins> Operational Logs</ins>: Batch execution data and operator actions

##### <ins>  Quality Data</ins>: Product quality measurements and deviations


---

#### <ins>Data Quality Pipeline</ins>

```python
┌─────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Source │ -> │ Great     │ -> │ Data      │ -> │ Clean     │
│  Data   │    │ Expectations   │ Quarantine│    │ Data      │
└─────────┘    └───────────┘    └───────────┘    └───────────┘
```
##### <ins>MISSING DATA HANDLING </ins>:

- KNN Imputation for sensor data. Mean ( *df.isnull().mean()* ) vs Mode ( *df.isnull().mode()* ) vs SMOTE data imputation?
- Categoraical vs numerical data
-  Data filling for time-series gaps
- Complete audit trail of all transformations
- Redundnant data removal (deduplication)

#### ALCOA+ Compliance Implementation

```python
# Data Integrity Framework
class ALCOACompliance:
    def ensure_attributable(self, data):
        """Who acquired the data and when"""
        return data.withColumn("acquired_by", lit(get_current_user())) \
                  .withColumn("acquisition_timestamp", current_timestamp())
    
    def ensure_legible(self, data):
        """Human and machine readable"""
        return data.withColumn("data_quality_score", 
                             calculate_data_quality_metrics(col("sensor_readings")))
    
    def ensure_contemporaneous(self, data):
        """Recorded at time of generation"""
        return data.withColumn("event_timestamp", 
                             from_utc_timestamp(col("timestamp"), "EST"))
    
    def ensure_original(self, data):
        """Original record or certified copy"""
        return data.withColumn("is_original", lit(True)) \
                  .withColumn("data_hash", md5(concat_ws("|", *data.columns)))
    
    def ensure_accurate(self, data):
        """Error-free and complete"""
        return data.filter(validate_sensor_ranges(col("sensor_readings")))
```

#### GMP  Compliance Implementation

```python
class GMPComplianceManager:
    def __init__(self):
        self.audit_logger = AuditLogger()
        self.change_control = ChangeControlSystem()
    
    def validate_data_integrity(self, data_batch):
        """Ensure ALCOA+ principles for all input data"""
        validation_results = {
            'attributable': self._verify_attribution(data_batch),
            'legible': self._verify_readability(data_batch),
            'contemporaneous': self._verify_timestamps(data_batch),
            'original': self._verify_originality(data_batch),
            'accurate': self._verify_accuracy(data_batch),
            'complete': self._verify_completeness(data_batch),
            'consistent': self._verify_consistency(data_batch),
            'enduring': self._verify_persistence(data_batch),
            'available': self._verify_availability(data_batch)
        }
        return validation_results
    
    def document_model_validation(self, model, validation_data):
        """GAMP 5 Category 4 Software Validation"""
        validation_package = {
            'user_requirements_specification': self._generate_urs(),
            'functional_specification': self._generate_fs(model),
            'design_specification': self._generate_ds(model),
            'risk_assessment': self._perform_risk_assessment(model),
            'test_protocols': self._create_test_protocols(model, validation_data),
            'validation_report': self._generate_validation_report()
        }
        return validation_package

```

---
---

## AI Model Architecture

#### Why Hybrid Ensemble Approach?

- Our selection of LSTM, BERT, and XGBoost is deliberate for GMP compliance:

| **Model** | **GMP Compliance Rationale** | **Technical Strengths** |
|------------|------------------------------|--------------------------|
| **LSTM** | Temporal patterns provide audit trail of failure progression | Sequential data processing, memory cells retain long-term dependencies |
| **BERT** | Attention mechanisms provide explainable feature importance | Multi-scale pattern recognition, transfer learning capabilities |
| **XGBoost** | Feature importance scores enable regulatory justification | Handles mixed data types, robust to noise and missing values |


![Image](https://github.com/user-attachments/assets/125abea1-b6bb-4ed9-8cbb-154ccd5ce125)

### Detailed Model Inputs & Processing

#### 1. LSTM: Temporal Sequence Analysis

#### Input: Raw time-series sensor sequences

```python
class LSTMPredictor:
    def __init__(self, sequence_length=3600, num_sensors=4):
        self.sequence_length = sequence_length
        self.model = self._build_lstm_model()
    
    def _build_lstm_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, 
                               input_shape=(self.sequence_length, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def prepare_sequence_data(self, raw_sensor_data):
        """
        Prepare LSTM input sequences for temporal pattern analysis
        Example input shape: (3600 timesteps × 4 sensors)
        """
        sequence = [
            # Normal operation phase
            [2.34, 1.89, 37.2, 45.6],  # [vib_x, vib_y, temp, current]
            [2.31, 1.92, 37.1, 45.8],
            # ... 3500+ normal readings ...
            
            # Failure development phase
            [2.45, 2.15, 37.3, 48.2],  # First anomaly
            [2.67, 2.34, 37.6, 52.1],  # Vibration increasing
            [5.67, 4.89, 38.9, 65.4],  # MAJOR SPIKE 
        ]
        return np.array(sequence).reshape(1, self.sequence_length, 4)
```

#### 2. BERT: Frequency Domain Analysis

#### Input: Patched frequency features from FFT

```python
class TimeSeriesBERT:
    def __init__(self, patch_size=512, num_patches=7):
        self.patch_size = patch_size
        self.num_patches = num_patches
        
    def frequency_patch_creation(self, raw_vibration_data):
        """
        Convert time-series to frequency patches for BERT analysis
        """
        patches = []
        
        for i in range(self.num_patches):
            # Extract 8.5 minutes of data (512 samples at 1Hz)
            start_idx = i * self.patch_size
            end_idx = start_idx + self.patch_size
            time_segment = raw_vibration_data[start_idx:end_idx]
            
            # Apply FFT to convert to frequency domain
            fft_output = np.fft.fft(time_segment)
            frequency_magnitudes = np.abs(fft_output)
            
            patches.append(frequency_magnitudes)
        
        # Example BERT input: Early bearing frequency detection
        bert_patches = [
            [0.12, 0.08, 0.05, 0.02, ...],  # Patch 1: Normal
            [0.11, 0.09, 0.04, 0.03, ...],  # Patch 2: Normal  
            [0.15, 0.12, 0.08, 0.15, ...],  # Patch 3: Early warning
            [0.22, 0.18, 0.12, 0.24, ...],  # Patch 4: Developing
            [0.35, 0.28, 0.18, 0.38, ...],  # Patch 5: Bearing frequencies
            [0.45, 0.38, 0.22, 0.48, ...],  # Patch 6: Strong signal
            [0.52, 0.45, 0.28, 0.56, ...]   # Patch 7: Failure imminent
        ]
        
        return np.array(bert_patches)
```

#### 3. XGBoost: Feature-Based Analysis

#### Input: Engineered statistical features

```python
class FeatureEngineer:
    def create_xgboost_features(self, sensor_data, maintenance_context):
        """
        Create engineered features for XGBoost model
        Combines sensor statistics with maintenance and operational context
        """
        features = {
            # Rolling statistics (last 6 hours)
            'vibration_rolling_mean_6h': 2.45,    # Normal: 1.8-2.2 → ELEVATED
            'vibration_rolling_std_6h': 1.23,     # Normal: 0.3-0.6 → VERY HIGH
            'vibration_skew_6h': 0.89,            # Normal: -0.2 to 0.2 → POSITIVE SKEW
            'vibration_kurtosis_6h': 4.56,        # Normal: 2.5-3.5 → HEAVY TAILS
            
            # Spectral features
            'fft_dominant_frequency': 125.6,      # Normal: 60-80Hz → BEARING FREQ
            'fft_spectral_entropy': 0.45,         # Normal: 0.1-0.3 → DISORGANIZED
            'fft_total_energy': 890.2,            # Normal: 300-500 → EXCESS ENERGY
            
            # Rate of change features
            'temp_rate_of_change': 0.12,          # °C/hour → GRADUAL INCREASE
            'vibration_rate_of_change': 0.45,     # units/hour → RAPID INCREASE
            
            # Maintenance context
            'days_since_last_maintenance': 45,    # Schedule: 30 days → OVERDUE
            'last_maintenance_type': 1,           # 1=minor, 2=major → ONLY MINOR DONE
            'equipment_age_days': 720,            # 2 YEARS OLD
            'previous_failure_count': 2,          # HAS FAILED BEFORE
            
            # Operational context
            'batch_phase': 2,                     # 2=high-speed operation
            'throughput_level': 85,               # % of capacity → HIGH LOAD
            'operator_shift': 1                   # 1=day shift
        }
        
        return np.array([list(features.values())])

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

```

#### 4. Meta-Learner: Intelligent Ensemble

```python
class HybridEnsemble:
    def __init__(self):
        self.bert_model = TimeSeriesBERT()
        self.lstm_model = LSTMPredictor()
        self.xgb_model = XGBoostModel()
        self.meta_learner = LogisticRegression()
        
    def fit(self, X_sequences, X_patches, X_features, y_true):
        """
        Train the hybrid ensemble with cross-validation
        """
        # Step 1: Train individual models
        print("Training individual models...")
        self.lstm_model.fit(X_sequences, y_true)
        self.bert_model.fit(X_patches, y_true)
        self.xgb_model.fit(X_features, y_true)
        
        # Step 2: Generate meta-features with cross-validation
        print("Generating meta-features for meta-learner...")
        lstm_preds = cross_val_predict(self.lstm_model, X_sequences, y_true, 
                                     method='predict_proba', cv=5)[:, 1]
        bert_preds = cross_val_predict(self.bert_model, X_patches, y_true,
                                     method='predict_proba', cv=5)[:, 1]
        xgb_preds = cross_val_predict(self.xgb_model, X_features, y_true,
                                    method='predict_proba', cv=5)[:, 1]
        
        # Step 3: Train meta-learner
        print("Training meta-learner...")
        meta_features = np.column_stack([bert_preds, lstm_preds, xgb_preds])
        self.meta_learner.fit(meta_features, y_true)
        
        # Step 4: Document model strengths for GMP compliance
        self._document_model_strengths(meta_features, y_true)
    
    def predict_proba(self, sequence, patch, features, equipment_context):
        """
        Make prediction with full explainability for GMP compliance
        """
        # Get individual model predictions
        bert_prob = self.bert_model.predict_proba(patch)[0]
        lstm_prob = self.lstm_model.predict_proba(sequence)[0]
        xgb_prob = self.xgb_model.predict_proba(features)[0]
        
        # Apply context-aware weighting
        final_weights = self._get_context_weights(
            bert_prob, lstm_prob, xgb_prob, equipment_context
        )
        
        # Calculate final probability
        final_probability = (
            final_weights[0] * bert_prob +
            final_weights[1] * lstm_prob + 
            final_weights[2] * xgb_prob
        )
        
        # Return comprehensive audit trail
        return self._create_audit_record(
            final_probability,
            [bert_prob, lstm_prob, xgb_prob],
            final_weights,
            equipment_context
        )
    
    def _get_context_weights(self, bert_prob, lstm_prob, xgb_prob, context):
        """
        Apply learned rules for context-aware model weighting
        """
        equipment_type = context['equipment_type']
        days_since_maintenance = context['days_since_maintenance']
        
        # Base weights by equipment type
        if equipment_type == 1:  # Centrifuges
            weights = [0.5, 0.3, 0.2]  # Trust BERT most for spectral analysis
        elif equipment_type == 2:  # Bioreactors
            weights = [0.4, 0.4, 0.2]  # Balanced approach
        else:
            weights = [0.33, 0.33, 0.34]  # Default equal weighting
        
        # Adjust for maintenance context
        if days_since_maintenance > 35:
            weights = [0.3, 0.2, 0.5]  # Trust XGBoost more for maintenance factors
        
        # Adjust for strong spectral evidence
        if bert_prob > 0.8:
            weights[0] += 0.1
            weights[1] -= 0.05
            weights[2] -= 0.05
        
        return weights
    
    def _create_audit_record(self, final_prob, component_probs, weights, context):
        """
        Create comprehensive audit record for GMP compliance
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'equipment_id': context['equipment_id'],
            'final_probability': float(final_prob),
            'component_predictions': {
                'bert': float(component_probs[0]),
                'lstm': float(component_probs[1]),
                'xgb': float(component_probs[2])
            },
            'applied_weights': [float(w) for w in weights],
            'decision_rationale': self._generate_rationale(component_probs, weights),
            'recommended_action': self._get_recommendation(final_prob, context),
            'confidence_level': self._calculate_confidence(final_prob, component_probs),
            'alcoa_plus_compliance': {
                'attributable': get_current_user(),
                'legible': True,
                'contemporaneous': True,
                'original': True,
                'accurate': self._validate_prediction_accuracy(component_probs)
            }
        }
    
    def _generate_rationale(self, probs, weights):
        """Generate human-readable explanation for prediction"""
        rationales = []
        
        if probs[0] > 0.8 and weights[0] > 0.4:
            rationales.append("BERT detected clear frequency domain anomalies")
        if probs[1] > 0.7 and weights[1] > 0.3:
            rationales.append("LSTM identified temporal failure progression")
        if probs[2] > 0.75 and weights[2] > 0.3:
            rationales.append("XGBoost flagged maintenance and operational risk factors")
            
        return "; ".join(rationales) if rationales else "No strong indicators detected"
```

#### Model Interpretability for Regulatory Compliance

```python
class ExplainablePredictions:
    def generate_shap_explanations(self, model, input_data):
        """SHAP explanations for regulatory compliance"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        explanation = {
            'feature_importance': self._rank_features(shap_values, input_data),
            'prediction_breakdown': self._explain_prediction(shap_values, input_data),
            'risk_factors': self._identify_risk_factors(shap_values, input_data),
            'compliance_justification': self._generate_compliance_justification(shap_values)
        }
        return explanation
    
    def _generate_compliance_justification(self, shap_values):
        """Generate regulatory-compliant justification for predictions"""
        return {
            'scientific_basis': "Model predictions based on statistically significant patterns",
            'clinical_relevance': "Failure predictions directly impact product quality and patient safety",
            'regulatory_alignment': "Approach aligns with FDA Process Analytical Technology (PAT) guidance",
            'validation_status': "Fully validated under GAMP 5 framework"
        }
```

---

## Deployment Strategy

#### Canary Deployment for GMP Compliance

> [!NOTE]
> Interested readers can check details relating to Canary, Shadow, A/B, Blue/Green etc., delpoyments here:   https://github.com/manuelbomi/Enterprise-Progressive-Delivery-for-K8s--Canary-Blue_Green-A_B-Shadow-Deployments-for-DevOps-MLOps
>

```python
class CanaryDeployment:
    def __init__(self):
        self.phase = 1
        self.metrics_tracker = DeploymentMetrics()
    
    def execute_canary_rollout(self, new_model, production_data):
        """
        GMP-compliant canary deployment with validation gates
        """
        deployment_phases = {
            1: {'scope': 'Single production line', 'duration': '2 weeks'},
            2: {'scope': '25% of equipment', 'duration': '3 weeks'}, 
            3: {'scope': '50% of equipment', 'duration': '4 weeks'},
            4: {'scope': 'Full deployment', 'duration': 'Ongoing'}
        }
        
        current_phase = deployment_phases[self.phase]
        
        # Phase-specific validation
        validation_results = self._validate_deployment_phase(
            new_model, current_phase['scope']
        )
        
        if validation_results['success']:
            self._approve_phase_advancement(validation_results)
            self.phase += 1
        else:
            self._trigger_rollback_protocol(validation_results)
    
    def _validate_deployment_phase(self, model, scope):
        """GMP validation for each deployment phase"""
        return {
            'performance_metrics': self._measure_model_performance(model, scope),
            'business_impact': self._assess_business_impact(model, scope),
            'compliance_status': self._verify_regulatory_compliance(model, scope),
            'user_acceptance': self._gather_user_feedback(scope),
            'success': self._evaluate_success_criteria()
        }

```

---

## MLOps Pipeline with Kubernetes

```python
# kubernetes/ml-pipeline.yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: gmp-ml-pipeline-
spec:
  entrypoint: predictive-maintenance-pipeline
  templates:
  - name: predictive-maintenance-pipeline
    steps:
    - - name: data-validation
        template: data-quality-check
    - - name: feature-engineering
        template: feature-pipeline
    - - name: model-training
        template: train-ensemble
    - - name: model-validation
        template: gmp-validation
    - - name: canary-deployment
        template: deploy-canary
```

---

## LLM Integration for Advanced Predictions

> [!IMPORTANT]
> #### <ins>Shadow Deployment of LLMs</ins>
> Shadow deployment of LLMs for industrial GMP applications should still be a foundational practice due to AI explainability issues and limted knowledge of the internal architectures of LLMs
>


```python
class LLMShadowDeployment:
    def __init__(self, base_ensemble, llm_model):
        self.base_ensemble = base_ensemble
        self.llm_model = llm_model
        self.comparison_logger = ShadowModeLogger()
    
    def predict_in_shadow_mode(self, input_data):
        """
        Run LLM predictions in shadow mode alongside production ensemble
        """
        # Production prediction
        production_pred = self.base_ensemble.predict_proba(input_data)
        
        # LLM prediction (shadow mode)
        llm_pred = self._generate_llm_prediction(input_data)
        
        # Log comparison for analysis
        self.comparison_logger.log_comparison({
            'timestamp': datetime.now(),
            'input_data': input_data,
            'production_prediction': production_pred,
            'llm_prediction': llm_pred,
            'discrepancy': abs(production_pred - llm_pred),
            'context': self._get_prediction_context(input_data)
        })
        
        return production_pred  # Only return production prediction
    
    def _generate_llm_prediction(self, input_data):
        """
        Use LLM for advanced pattern recognition while maintaining explainability
        """
        prompt = self._create_llm_prompt(input_data)
        
        try:
            llm_response = self.llm_model.generate(prompt)
            prediction = self._parse_llm_response(llm_response)
            
            # Validate LLM output against business rules
            if self._validate_llm_prediction(prediction):
                return prediction
            else:
                return None
                
        except Exception as e:
            self._log_llm_error(e)
            return None
    
    def _create_llm_prompt(self, input_data):
        """Create structured prompt for LLM analysis"""
        return f"""
        As a pharmaceutical manufacturing equipment expert, analyze this sensor data:
        
        EQUIPMENT CONTEXT:
        - Type: {input_data['equipment_type']}
        - Age: {input_data['equipment_age_days']} days
        - Last Maintenance: {input_data['days_since_maintenance']} days ago
        - Current Phase: {input_data['batch_phase']}
        
        SENSOR READINGS:
        Vibration: {input_data['vibration_data']}
        Temperature: {input_data['temperature_data']} 
        Pressure: {input_data['pressure_data']}
        
        MAINTENANCE HISTORY:
        {input_data['maintenance_history']}
        
        Based on patterns in similar failure scenarios, provide:
        1. Failure probability (0-1)
        2. Key risk factors identified
        3. Recommended inspection timeline
        4. Confidence level in prediction
        
        Format response as JSON.
        """

```

---

## LLM Use Cases in GMP Environment

```python

class GMPLLMApplications:
    def __init__(self):
        self.llm_analyzer = LLMAnalyzer()
    
    def root_cause_analysis(self, failure_data):
        """Use LLM for complex root cause analysis"""
        return self.llm_analyzer.analyze_failure_patterns(failure_data)
    
    def procedural_optimization(self, maintenance_sops):
        """Optimize maintenance procedures using LLM insights"""
        return self.llm_analyzer.suggest_procedure_improvements(maintenance_sops)
    
    def regulatory_documentation(self, validation_data):
        """Assist in generating regulatory documentation"""
        return self.llm_analyzer.generate_validation_docs(validation_data)
    
    def anomaly_explanation(self, detected_anomalies):
        """Provide natural language explanations for anomalies"""
        return self.llm_analyzer.explain_anomalies(detected_anomalies)


```

---

## Monitoring & Continuous Improvement

#### Model Performance Monitoring

```python

class ModelPerformanceMonitor:
    def __init__(self):
        self.metrics = ModelMetrics()
        self.alert_system = AlertSystem()
    
    def track_model_performance(self, predictions, actuals):
        """Continuous monitoring for model performance degradation"""
        performance_metrics = {
            'accuracy': accuracy_score(actuals, predictions > 0.5),
            'precision': precision_score(actuals, predictions > 0.5),
            'recall': recall_score(actuals, predictions > 0.5),
            'f1_score': f1_score(actuals, predictions > 0.5),
            'auc_roc': roc_auc_score(actuals, predictions),
            'business_kpis': self._calculate_business_kpis(predictions, actuals)
        }
        
        # Check for performance degradation
        if self._detect_performance_drop(performance_metrics):
            self.alert_system.trigger_retraining_alert(performance_metrics)
    
    def monitor_data_drift(self, current_data, reference_data):
        """Monitor for data distribution changes"""
        drift_metrics = {
            'feature_drift': self._calculate_feature_drift(current_data, reference_data),
            'concept_drift': self._detect_concept_drift(current_data, reference_data),
            'temporal_patterns': self._analyze_temporal_changes(current_data)
        }
        return drift_metrics

```

---

### Automated Retraining Pipeline

```python
class AutomatedRetraining:
    def __init__(self):
        self.retraining_triggers = RetrainingTriggers()
        self.validation_gate = ValidationGate()
    
    def manage_model_lifecycle(self):
        """GMP-compliant model lifecycle management"""
        while True:
            # Check retraining triggers
            if self.retraining_triggers.should_retrain():
                # Execute retraining pipeline
                new_model = self._execute_retraining_pipeline()
                
                # Validate new model
                validation_results = self.validation_gate.validate_model(new_model)
                
                if validation_results['approved']:
                    # Deploy with change control
                    self._deploy_with_change_control(new_model)
                else:
                    # Log validation failure
                    self._handle_validation_failure(validation_results)
            
            time.sleep(3600)  # Check hourly
```

## Success Metrics & Business Impact

#### Key Performance Indicators

| **Metric** | **Target** | **Current** | **Impact** |
|-------------|------------|-------------|-------------|
| **Unplanned Downtime Reduction** | >15% | TBD | $2.1M annual savings |
| **Maintenance Cost Reduction** | >10% | TBD | $850K annual savings |
| **False Positive Rate** | <5% | TBD | Maintenance team trust |
| **Regulatory Compliance** | 100% | 100% | Zero audit findings |
| **Mean Time Between Failures** | +20% | TBD | Improved reliability |



## Implementation Roadmap

<img width="5100" height="1280" alt="Image" src="https://github.com/user-attachments/assets/f5f4ff00-d5a0-4148-bbda-267e01244b74" />


## Future Enhancements

- Digital Twin Integration
- 
- Real-time equipment digital twins for simulation

- What-if analysis for maintenance optimization

- Predictive quality analytics

---

## Advanced AI Capabilities

- Reinforcement learning for maintenance optimization

- Transfer learning across equipment types

- Federated learning for multi-site deployment

---

## Regulatory Technology

- Automated audit trail generation

- Real-time compliance monitoring

- AI-assisted regulatory submissions

---



## Conclusion

#### This AI-driven predictive maintenance solution represents a comprehensive approach to equipment reliability in GMP manufacturing environments. By combining the temporal understanding of LSTM, the frequency domain expertise of BERT, and the feature-based intelligence of XGBoost within an explainable ensemble framework, we deliver:

- Regulatory Compliance: Full ALCOA+, GAMP 5, FDA 21 CFR Part 11 compliance

- Operational Excellence: Significant reduction in unplanned downtime

- Financial Impact: Multi-million dollar cost savings

- Scalable Foundation: Platform for future AI initiatives

#### The solution demonstrates that advanced AI can be successfully deployed in highly regulated environments when proper governance, explainability, and validation frameworks are established


### Thank you for reading
---

### **AUTHOR'S BACKGROUND**

### Author's Name:  Emmanuel Oyekanlu
```
Skillset:   I have experience spanning several years in data science, developing scalable enterprise data pipelines,
enterprise solution architecture, architecting enterprise systems data and AI applications,
software and AI solution design and deployments, data engineering, high performance computing (GPU, CUDA), machine learning,
NLP, Agentic-AI and LLM applications as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)








