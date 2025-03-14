import logging
from typing import Dict, List, Tuple, Optional, Union
import pulp
from dream11.config import DEFAULT_CONSTRAINTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TeamSelector:
    """
    Team selector module for selecting the optimal team based on constraints and predicted points.
    Uses PuLP (Linear Programming) to solve the optimization problem.
    """
    
    def __init__(self, db, ml_model):
        """
        Initialize the TeamSelector object.
        
        Args:
            db: Database object for player information
            ml_model: ML model for predicting player points
        """
        self.db = db
        self.ml_model = ml_model
        self.constraints = DEFAULT_CONSTRAINTS.copy()
        self.selected_team = []
        self.player_points = {}
        
    def update_constraints(self, constraints: Dict) -> None:
        """
        Update team selection constraints.
        
        Args:
            constraints: Dictionary of constraints to update
        """
        for key, value in constraints.items():
            if key in self.constraints:
                self.constraints[key] = value
                
        logger.info(f"Updated constraints: {constraints}")
    
    def select_team(self, team1: str, team2: str, 
                   specific_players: List[str] = None, 
                   excluded_players: List[str] = None,
                   favor_spinners: bool = None,
                   team1_players: int = None) -> List[Dict]:
        """
        Select an optimal team of 11 players based on constraints and predicted points.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            specific_players: List of player names that must be included
            excluded_players: List of player names that must be excluded
            favor_spinners: Whether to favor spinners in selection
            team1_players: Number of players to select from team1
            
        Returns:
            List of selected player dictionaries
        """
        logger.info(f"Selecting team for match: {team1} vs {team2}")
        
        # Create player pool
        player_pool = self.db.create_player_pool(team1, team2)
        
        if not player_pool:
            logger.error(f"No players found for teams {team1} and {team2}")
            return []
            
        # Get predicted points for each player
        venue = "neutral"  # Default venue, can be updated based on match data
        city = "unknown"   # Default city, can be updated based on match data
        
        # In a real implementation, you would get the venue and city from match data
        # For simplicity, we'll use defaults
        
        predicted_points = self.ml_model.predict_match_points(team1, team2, venue, city)
        
        # Store for later use
        self.player_points = predicted_points
        
        # Set up PuLP problem
        prob = pulp.LpProblem("Dream11_Team_Selection", pulp.LpMaximize)
        
        # Create binary variables for each player
        player_vars = {}
        for player in player_pool:
            player_name = player['name']
            player_vars[player_name] = pulp.LpVariable(f"select_{player_name}", cat='Binary')
        
        # Objective function: maximize total points
        prob += pulp.lpSum([player_vars[player] * predicted_points.get(player, 0) for player in player_vars])
        
        # Constraint: exactly 11 players
        prob += pulp.lpSum([player_vars[player] for player in player_vars]) == self.constraints['TOTAL_PLAYERS']
        
        # Constraint: credit limit
        prob += pulp.lpSum([player_vars[player] * next((p['credits'] for p in player_pool if p['name'] == player), 0) 
                           for player in player_vars]) <= self.constraints['TOTAL_CREDITS']
        
        # Role constraints
        # Get players by role
        batsmen = [p['name'] for p in player_pool if p['role'] == 'batsman']
        bowlers = [p['name'] for p in player_pool if p['role'] == 'bowler']
        all_rounders = [p['name'] for p in player_pool if p['role'] == 'all-rounder']
        wicket_keepers = [p['name'] for p in player_pool if p['role'] == 'wicket-keeper']
        
        # Batsmen constraint
        prob += pulp.lpSum([player_vars[player] for player in batsmen]) >= self.constraints['BATSMEN']['MIN']
        prob += pulp.lpSum([player_vars[player] for player in batsmen]) <= self.constraints['BATSMEN']['MAX']
        
        # Bowlers constraint
        prob += pulp.lpSum([player_vars[player] for player in bowlers]) >= self.constraints['BOWLERS']['MIN']
        prob += pulp.lpSum([player_vars[player] for player in bowlers]) <= self.constraints['BOWLERS']['MAX']
        
        # All-rounders constraint
        prob += pulp.lpSum([player_vars[player] for player in all_rounders]) >= self.constraints['ALL_ROUNDERS']['MIN']
        prob += pulp.lpSum([player_vars[player] for player in all_rounders]) <= self.constraints['ALL_ROUNDERS']['MAX']
        
        # Wicket-keepers constraint
        prob += pulp.lpSum([player_vars[player] for player in wicket_keepers]) >= self.constraints['WICKET_KEEPERS']['MIN']
        prob += pulp.lpSum([player_vars[player] for player in wicket_keepers]) <= self.constraints['WICKET_KEEPERS']['MAX']
        
        # Team distribution constraint (if specified)
        if team1_players is not None:
            team1_players_list = [p['name'] for p in player_pool if p['team'] == team1]
            team2_players_list = [p['name'] for p in player_pool if p['team'] == team2]
            
            prob += pulp.lpSum([player_vars[player] for player in team1_players_list]) == team1_players
            prob += pulp.lpSum([player_vars[player] for player in team2_players_list]) == (self.constraints['TOTAL_PLAYERS'] - team1_players)
        else:
            # Default constraint on max players from a team
            team1_players_list = [p['name'] for p in player_pool if p['team'] == team1]
            prob += pulp.lpSum([player_vars[player] for player in team1_players_list]) <= self.constraints['MAX_PLAYERS_FROM_TEAM']
            
            team2_players_list = [p['name'] for p in player_pool if p['team'] == team2]
            prob += pulp.lpSum([player_vars[player] for player in team2_players_list]) <= self.constraints['MAX_PLAYERS_FROM_TEAM']
        
        # Spinner preference constraint (if specified)
        if favor_spinners is not None:
            spinners = [p['name'] for p in player_pool if p['is_spinner']]
            pacers = [p['name'] for p in player_pool if p['role'] == 'bowler' and not p['is_spinner']]
            
            if favor_spinners:
                # Constraint: more spinners than pacers
                prob += pulp.lpSum([player_vars[player] for player in spinners]) >= pulp.lpSum([player_vars[player] for player in pacers])
            else:
                # Constraint: more pacers than spinners
                prob += pulp.lpSum([player_vars[player] for player in spinners]) <= pulp.lpSum([player_vars[player] for player in pacers])
        
        # Include specific players (if specified)
        if specific_players:
            for player in specific_players:
                if player in player_vars:
                    prob += player_vars[player] == 1
                    logger.info(f"Forcing inclusion of player: {player}")
        
        # Exclude specific players (if specified)
        if excluded_players:
            for player in excluded_players:
                if player in player_vars:
                    prob += player_vars[player] == 0
                    logger.info(f"Forcing exclusion of player: {player}")
        
        # Solve the problem
        logger.info("Solving team selection optimization problem...")
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Check if a solution was found
        if pulp.LpStatus[prob.status] != 'Optimal':
            logger.error(f"No optimal solution found. Status: {pulp.LpStatus[prob.status]}")
            return []
        
        # Extract selected players
        selected_players = []
        total_points = 0
        total_credits = 0
        
        for player in player_pool:
            player_name = player['name']
            if player_name in player_vars and player_vars[player_name].value() == 1:
                player_points = predicted_points.get(player_name, 0)
                player_info = player.copy()
                player_info['predicted_points'] = player_points
                selected_players.append(player_info)
                
                total_points += player_points
                total_credits += player['credits']
        
        # Sort by predicted points (descending)
        selected_players.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Store the selected team
        self.selected_team = selected_players
        
        logger.info(f"Team selection complete: {len(selected_players)} players selected")
        logger.info(f"Total predicted points: {total_points:.2f}, Total credits: {total_credits}")
        
        return selected_players
    
    def suggest_captain_vice_captain(self) -> Tuple[Dict, Dict]:
        """
        Suggest captain and vice-captain from the selected team.
        
        Returns:
            Tuple of (captain, vice_captain) dictionaries
        """
        if not self.selected_team:
            logger.error("No team has been selected yet")
            return None, None
        
        # Simply choose the two players with highest predicted points
        sorted_team = sorted(self.selected_team, key=lambda x: x['predicted_points'], reverse=True)
        
        captain = sorted_team[0]
        vice_captain = sorted_team[1]
        
        logger.info(f"Suggested captain: {captain['name']} ({captain['predicted_points']:.2f} points)")
        logger.info(f"Suggested vice-captain: {vice_captain['name']} ({vice_captain['predicted_points']:.2f} points)")
        
        return captain, vice_captain
    
    def get_team_summary(self) -> Dict:
        """
        Get a summary of the selected team.
        
        Returns:
            Dictionary with team summary
        """
        if not self.selected_team:
            logger.error("No team has been selected yet")
            return {}
        
        # Count players by role
        roles = {'batsman': 0, 'bowler': 0, 'all-rounder': 0, 'wicket-keeper': 0}
        teams = {}
        total_credits = 0
        total_points = 0
        
        for player in self.selected_team:
            roles[player['role']] += 1
            
            if player['team'] in teams:
                teams[player['team']] += 1
            else:
                teams[player['team']] = 1
                
            total_credits += player['credits']
            total_points += player['predicted_points']
        
        # Get captain and vice-captain
        captain, vice_captain = self.suggest_captain_vice_captain()
        
        summary = {
            'total_players': len(self.selected_team),
            'total_credits': total_credits,
            'total_points': total_points,
            'roles': roles,
            'teams': teams,
            'captain': captain['name'] if captain else None,
            'vice_captain': vice_captain['name'] if vice_captain else None
        }
        
        return summary
    
    def get_selected_team(self) -> List[Dict]:
        """
        Get the selected team.
        
        Returns:
            List of selected player dictionaries
        """
        return self.selected_team 