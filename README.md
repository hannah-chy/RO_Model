# RO_Model
RO_Retention_Prediction_Model

## Project Overview
This is a machine learning framework for analyzing and visualizing the contributions of different mechanisms in separation processes. The framework can evaluate the impact of mechanisms such as electrostatic interactions, steric effects, and intermolecular forces on separation processes, providing detailed visualization analysis results.

## Key Features
- Multi-mechanism contribution analysis
- Automated feature engineering
- Interaction effect assessment
- Visualization of results
- Comprehensive experiment logging system

## Project Structure
RO_Model/
├── main.py # Main program entry
├── data_processor.py # Data processing module
├── model_trainer.py # Model training module
├── mechanism_contributions.py # Mechanism contribution analysis module
├── visualizer.py # Visualization module
└── experiment_logger.py # Experiment logging module

## Installation Dependencies
bash
pip install -r requirements.txt

## Usage
1. Prepare data file (CSV format)
2. Configure parameters (in main.py)
3. Run analysis:

## Usage
1. Prepare data file (CSV format)
2. Configure parameters (in main.py)
3. Run analysis:
python
python main.py

## Configuration Details

python
config = {
'file_path': 'data file path',
'base_output_dir': 'output directory',
'features': ['MWCO', 'contact_angle', 'pH', 'pres', 'initial_conc', # ... other features],
'target': 'rejection',
'group_methods': ['none', 'cmpd', 'ref'],
'feature_methods': ['original', 'standard', # ... other features]
}

## Author
[Han-Ying Cai]

## Contributions
Issues and Pull Requests are welcome.

## Citation
If you use this framework in your research, please cite:
[Paper citation information]
