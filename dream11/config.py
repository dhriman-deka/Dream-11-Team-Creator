import os

# File paths
MATCHES_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matches.csv')
DELIVERIES_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deliveries.csv')

# Dream 11 points system
POINTS_SYSTEM = {
    'BATTING': {
        'RUN': 1,  # 1 point per run
        'BOUNDARY_BONUS': {
            '4': 1,  # 1 extra point for a four
            '6': 2,  # 2 extra points for a six
        },
        'MILESTONE_BONUS': {
            '30': 4,   # 4 bonus points for 30+ runs
            '50': 12,   # 8 bonus points for 50+ runs
            '100': 16, # 16 bonus points for 100+ runs
        },
        'DISMISSAL_FOR_DUCK': -2,  # -2 points for duck (only for batsmen)
    },
    'BOWLING': {
        'WICKET': 25,  # 25 points per wicket (excluding run-out)
        'BONUS_WICKETS': {
            '3': 4,   # 4 bonus points for 3 wickets
            '4': 8,   # 8 bonus points for 4 wickets
            '5': 16,  # 16 bonus points for 5 wickets
        },
        'MAIDEN_OVER': 8,  # 8 points per maiden over
        'ECONOMY_RATE': {
            # Economy rate bonus/penalty
            'LESS_THAN_5': 6,    # 6 bonus points for economy rate < 5
            'BETWEEN_5_6': 4,    # 4 bonus points for economy rate 5-6
            'BETWEEN_6_7': 2,    # 2 bonus points for economy rate 6-7
            'BETWEEN_10_11': -2, # -2 penalty points for economy rate 10-11
            'BETWEEN_11_12': -4, # -4 penalty points for economy rate 11-12
            'MORE_THAN_12': -6,  # -6 penalty points for economy rate > 12
        }
    },
    'FIELDING': {
        'CATCH': 8,     # 8 points per catch
        'STUMPING': 12, # 12 points per stumping
        'RUN_OUT': {
            'DIRECT_HIT': 12,    # 12 points for direct hit run out
            'INDIRECT_HIT': 6,   # 6 points for indirect run out
        }
    },
    'OTHER': {
        'CAPTAIN_MULTIPLIER': 4,      # Captain points multiplied by 2
        'VICE_CAPTAIN_MULTIPLIER': 3.5  # Vice-captain points multiplied by 1.5
    }
}

# Default team constraints
DEFAULT_CONSTRAINTS = {
    'TOTAL_PLAYERS': 11,
    'TOTAL_CREDITS': 100,
    'BATSMEN': {
        'MIN': 2,
        'MAX': 5
    },
    'BOWLERS': {
        'MIN': 2,
        'MAX': 5
    },
    'ALL_ROUNDERS': {
        'MIN': 0,
        'MAX': 4
    },
    'WICKET_KEEPERS': {
        'MIN': 1,
        'MAX': 2
    },
    'MAX_PLAYERS_FROM_TEAM': 7
}

# ML model parameters
ML_CONFIG = {
    'TEST_YEAR': 2019,   # Use 2019 for test data
    'TRAIN_YEARS': list(range(2015, 2019)),  # Use fewer years for training (2015-2018 instead of 2008-2018)
    'FEATURES': [
        'total_runs_scored', 
        'total_balls_faced', 
        'no_4s', 
        'no_6s', 
        'runs_conceded', 
        'overs_bowled', 
        'no_wickets_taken',
        'average_runs', 
        'strike_rate', 
        'bowling_average', 
        'economy_rate', 
        'player_consistency',
        'fielding_score',
        'venue', 
        'city', 
        'opposition_team'
    ],
    'CATBOOST_PARAMS': {
        'iterations': 300,  # Reduced from 1000
        'learning_rate': 0.05,  # Increased from 0.03
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 100,
        'early_stopping_rounds': 50  # Added early stopping
    },
    'XGBOOST_PARAMS': {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbose_eval': 100,
        'num_boost_round': 300,  # Reduced from 1000
        'early_stopping_rounds': 50  # Added early stopping
    },
    'RANDOM_FOREST_PARAMS': {
        'n_estimators': 50,  # Reduced from 100
        'max_depth': 8,      # Reduced from 10
        'min_samples_split': 5,  # Increased from 2 for faster training
        'min_samples_leaf': 2,   # Increased from 1 for faster training
        'random_state': 42,
        'n_jobs': -1  # Use all available cores
    }
}
