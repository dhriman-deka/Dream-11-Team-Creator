# Dream 11 Team Creator

An intelligent cricket team selection system using machine learning to create optimal Dream 11 fantasy cricket teams based on real player statistics.

## Overview

Dream 11 Team Creator helps cricket enthusiasts create winning fantasy cricket teams by analyzing player performance data using advanced machine learning algorithms. The system considers historical player statistics, venue conditions, team matchups, and user-defined constraints to recommend the best possible team selection.

![Dream 11 Team Creator Screenshot](screenshots/app_screenshot.png)
*Note: Add a screenshot of your application to the screenshots directory before pushing to GitHub*

## Key Features

- **AI-Powered Team Selection**: Uses ensemble machine learning models (CatBoost, XGBoost, Random Forest) to predict player performance
- **Team Optimization**: Employs linear programming to select the optimal combination of players within given constraints
- **Customizable Constraints**: Set your own criteria for team selection (number of batsmen, bowlers, all-rounders, etc.)
- **Player Statistics**: Detailed performance metrics and form analysis for all players
- **Captain & Vice-Captain Selection**: ML-based recommendations for captain and vice-captain based on expected performance
- **Interactive Web Interface**: User-friendly dashboard for team creation and analysis
- **Team Export**: Export your selected teams as JSON files for future reference

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Optimization**: PuLP (Linear Programming)
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data Analysis**: Pandas, NumPy

## Installation & Setup

### Prerequisites
- Python 3.8+
- IPL match data (CSV files)
- 500MB+ free disk space for models and data

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dream11-team-creator.git
cd dream11-team-creator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Place data files in the root directory:
- `matches.csv`: Match-level data for IPL
- `deliveries.csv`: Ball-by-ball data for IPL matches

5. Start the application:
```bash
python -m dream11.app
```

6. Access the web interface at http://localhost:5001

## How It Works

1. **Data Preparation**: The system processes IPL match data, extracting player statistics and performance metrics.

2. **Machine Learning**: Multiple regression models are trained to predict player performance in upcoming matches.

3. **Optimization**: Linear programming is used to select the best 11 players within constraints like:
   - Total credits (100)
   - Team composition (batsmen, bowlers, all-rounders, wicket-keepers)
   - Maximum players from a single team (7)

4. **Model Caching**: Team-specific models are cached to reduce training time for repeated team combinations.

## Performance Optimization

The application includes several performance optimizations:
- Team-specific model caching to avoid retraining
- Efficient feature engineering to reduce dimensionality
- Early stopping during model training
- Parallelized training using available CPU cores

## Project Structure

```
dream11-team-creator/
├── dream11/                  # Main application package
│   ├── app.py                # Flask application
│   ├── config.py             # Configuration settings
│   ├── precompute_models.py  # Model precomputation utilities
│   ├── modules/              # Application modules
│   │   ├── database/         # Database handling
│   │   ├── input/            # Input processing
│   │   ├── ml/               # Machine learning models
│   │   └── selection/        # Team selection algorithms
│   ├── static/               # Static assets (CSS, JS)
│   └── templates/            # HTML templates
├── models/                   # Trained ML models
│   └── team_specific/        # Team-specific cached models
├── exports/                  # Exported team data
├── matches.csv               # Match data
├── deliveries.csv            # Ball-by-ball data
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # Project documentation
```

## Future Enhancements

- Real-time data integration with live matches
- Player form trend analysis
- Support for multiple cricket formats (T20, ODI, Test)
- Mobile application development
- Collaborative team creation

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IPL for providing the cricket match data
- The open-source machine learning community
- Fantasy cricket enthusiasts for their valuable feedback 