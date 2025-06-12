# California Wine Industry Environmental Impact Analyzer 🍷

## Overview

The California Wine Industry Environmental Impact Analyzer is a comprehensive data analytics and visualization platform that helps stakeholders understand and analyze the environmental impacts on California's wine industry. This interactive web application provides real-time insights into various factors affecting wine production, quality, and sustainability across California's major wine regions.

## Features

### 1. Regional Analysis
- Interactive maps of California wine regions
- Detailed statistics for major wine-producing areas:
  - Napa Valley
  - Sonoma County
  - Paso Robles
  - Santa Barbara
  - Mendocino County
  - Lodi

### 2. Environmental Impact Tracking
- Wildfire impact analysis and visualization
- Weather pattern monitoring
- Drought impact assessment
- Smoke exposure analysis for vineyards

### 3. Production Analytics
- Historical production data analysis
- Quality score tracking
- Revenue impact assessment
- Regional comparison tools

### 4. Predictive Modeling
- Machine learning-based predictions for:
  - Future production trends
  - Quality forecasting
  - Environmental risk assessment

### 5. Interactive Visualizations
- Dynamic maps using Folium
- Time-series analysis with Plotly
- Statistical visualizations with Seaborn
- Custom-styled Streamlit components

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd CaWine
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

## Dependencies

The application relies on several Python packages:
- streamlit>=1.27.0 - Web application framework
- pandas>=1.5.0 - Data manipulation and analysis
- numpy>=1.24.0 - Numerical computing
- matplotlib>=3.7.0 - Data visualization
- seaborn>=0.12.0 - Statistical data visualization
- plotly>=5.15.0 - Interactive plots
- folium>=0.14.0 - Interactive maps
- streamlit-folium>=0.13.0 - Folium integration for Streamlit
- scikit-learn>=1.2.0 - Machine learning functionality
- geopandas>=0.13.0 - Geospatial data operations
- requests>=2.31.0 - HTTP library
- statsmodels>=0.14.4 - Statistical models

## Data Sources

The application uses various data sources:
- Historical wildfire data from CAL FIRE
- Weather data from NOAA
- Wine production statistics from California wine regions
- Geographic data for wine regions
- Environmental impact assessments

## Features in Detail

### Wildfire Impact Analysis
- Tracks historical wildfire events affecting wine regions
- Calculates smoke exposure risk
- Visualizes fire proximity to vineyards
- Assesses impact on wine quality

### Weather Pattern Monitoring
- Temperature trends analysis
- Rainfall patterns
- Extreme weather event tracking
- Climate change impact assessment

### Production Analytics
- Year-over-year production comparisons
- Quality score tracking
- Revenue impact analysis
- Regional performance metrics

### Predictive Analytics
- Machine learning models for production forecasting
- Quality prediction based on environmental factors
- Risk assessment for future seasons

## Contributing

Contributions to improve the analyzer are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify License]

## Contact

[Specify Contact Information]

## Acknowledgments

- California Wine Institute
- CAL FIRE
- NOAA
- [Other relevant organizations]