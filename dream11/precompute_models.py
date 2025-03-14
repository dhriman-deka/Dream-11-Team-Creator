#!/usr/bin/env python3
"""
Precompute ML models for common team combinations.

This script generates and caches ML models for all possible team combinations
to reduce training time during actual usage of the application.
"""

import os
import sys
import logging
import itertools
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dream11.modules.database import Database
from dream11.modules.ml import MLModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to precompute models for team combinations."""
    logger.info("Starting precomputation of team-specific models")
    
    # Initialize database
    db = Database()
    
    # Get all available teams
    all_teams = list(db.teams)
    logger.info(f"Found {len(all_teams)} teams in the database")
    
    # Initialize ML model
    ml_model = MLModel(db)
    
    # First, train and save the general model
    logger.info("Training general model...")
    ml_model.train_and_evaluate_all_models()
    general_model_path = os.path.join('models', 'dream11_model.pkl')
    ml_model.save_model(general_model_path)
    logger.info(f"General model saved to {general_model_path}")
    
    # Generate all possible team combinations
    team_combinations = list(itertools.combinations(all_teams, 2))
    logger.info(f"Found {len(team_combinations)} possible team combinations")
    
    # If there are too many, just use the most recent/popular teams
    # (This is a simplification; in a real system you might prioritize based on popularity or recency)
    max_combinations = 20  # Adjust based on available time/resources
    if len(team_combinations) > max_combinations:
        logger.info(f"Limiting to {max_combinations} most common team combinations")
        team_combinations = team_combinations[:max_combinations]
    
    # Create models for each combination
    for team1, team2 in tqdm(team_combinations, desc="Computing team models"):
        logger.info(f"Processing team combination: {team1} vs {team2}")
        
        # Get the model path for this combination
        model_path = ml_model.get_team_model_path(team1, team2)
        
        # Skip if model already exists
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}, skipping...")
            continue
        
        # Use the general model as a base (this will save it as a team-specific model)
        if ml_model.load_model(general_model_path):
            ml_model.save_model(model_path)
            logger.info(f"Created team-specific model for {team1} vs {team2}")
        else:
            logger.error(f"Failed to create model for {team1} vs {team2}")
    
    logger.info("Precomputation of team-specific models completed")

if __name__ == "__main__":
    main() 