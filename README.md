# tec-mlops-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

# **MLOps Project: Team 2.**

This project implements a complete **MLOps pipeline** to predict the number of Bike Sharing.
It uses **DVC** for tracking data, and **MLflow** to manage experiments and the model lifecycle. 
The pipeline includes data preparation, model training, and evaluation.

## **Project Structure**

```bash
tec-mlops-project/
├── .dvc/                   
├── data
│   ├── external       <- Data from third-party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical datasets for modeling.
│   └── raw            <- The original, immutable data dump.       
├── models/                 
├── src/
│   ├── stages/           
│   │   ├── data_clean.py          
│   │   ├── data_load.py          
│   │   ├── preprocess.py   
│   │   ├── split.py   
│   │   └── train.py   
│   ├── utils/                 
│   │   ├── dataExplorer.py          
│   │   └── utils.py      
├── notebooks/
├── modeling_pipeline/    <- MLflow pipeline
│   ├── docker-compose.yaml
│   └── mlflow/
├── setup.cfg          
├── tec_mlops_project/   
│   ├── main.py             
│   └── bikeSharingModel.py                
└── dvc.yaml    
```

## **Setup and Running**

### **Prerequisites**

- Docker & Docker Compose
- Python 3.8+ and `pip`
- DVC (`pip install dvc`)
- **Remote storage** set up for DVC (e.g., S3 or GDrive)

### **Clone the Repository**

```bash
git clone https://github.com/juanricardoab/tec-mlops-project.git
cd tec-mlops-project
```

### **Virtual Environment Setup**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **Run the Project with Docker Compose**

#### **Development**:

```bash
cd modeling_pipeline
docker compose --env-file config.env up -d --build
```


