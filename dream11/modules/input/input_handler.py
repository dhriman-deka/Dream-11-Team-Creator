import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InputHandler:
    """
    Input Handler class to manage user inputs for Dream 11 Team Creator.
    """
    
    def __init__(self, db):
        """
        Initialize the InputHandler object.
        
        Args:
            db: Database object for accessing team and player data
        """
        self.db = db
        self.selected_teams = []
        self.constraints = {}
        self.specific_players = []
        self.excluded_players = []
        self.favor_spinners = None
        self.team1_players = None
        
    def get_available_teams(self) -> List[str]:
        """
        Get a list of all available teams.
        
        Returns:
            List of team names
        """
        return list(self.db.teams)
    
    def select_teams(self, team1: str, team2: str) -> bool:
        """
        Select two teams for the match.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            
        Returns:
            Boolean indicating success
        """
        if team1 not in self.db.teams or team2 not in self.db.teams:
            logger.error(f"One or both teams not found: {team1}, {team2}")
            return False
        
        if team1 == team2:
            logger.error("Cannot select the same team twice")
            return False
        
        self.selected_teams = [team1, team2]
        logger.info(f"Selected teams: {team1} vs {team2}")
        
        return True
    
    def set_constraints(self, constraints: Dict) -> None:
        """
        Set constraints for team selection.
        
        Args:
            constraints: Dictionary of constraints
        """
        self.constraints = constraints
        logger.info(f"Set constraints: {constraints}")
    
    def set_specific_players(self, player_names: List[str]) -> None:
        """
        Set specific players to include in the team.
        
        Args:
            player_names: List of player names to include
        """
        self.specific_players = player_names
        logger.info(f"Set specific players to include: {player_names}")
    
    def set_excluded_players(self, player_names: List[str]) -> None:
        """
        Set specific players to exclude from the team.
        
        Args:
            player_names: List of player names to exclude
        """
        self.excluded_players = player_names
        logger.info(f"Set specific players to exclude: {player_names}")
    
    def set_spinner_preference(self, favor_spinners: bool) -> None:
        """
        Set preference for spinners.
        
        Args:
            favor_spinners: Whether to favor spinners (True) or pace bowlers (False)
        """
        self.favor_spinners = favor_spinners
        logger.info(f"Set spinner preference: {'Favor spinners' if favor_spinners else 'Favor pace bowlers'}")
    
    def set_team_distribution(self, team1_players: int) -> bool:
        """
        Set the number of players to select from team1.
        
        Args:
            team1_players: Number of players to select from team1
            
        Returns:
            Boolean indicating success
        """
        if not self.selected_teams:
            logger.error("No teams selected yet")
            return False
        
        if team1_players < 0 or team1_players > 11:
            logger.error("Invalid number of players (must be between 0 and 11)")
            return False
        
        self.team1_players = team1_players
        logger.info(f"Set team distribution: {team1_players} players from {self.selected_teams[0]}, {11 - team1_players} players from {self.selected_teams[1]}")
        
        return True
    
    def get_player_details(self, player_name: str) -> Dict:
        """
        Get details for a specific player.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dictionary with player details
        """
        player_performance = self.db.get_player_performance(player_name)
        
        if not player_performance:
            logger.error(f"Player not found: {player_name}")
            return {}
        
        return {
            'name': player_name,
            'performance': player_performance
        }
    
    def get_all_inputs(self) -> Dict:
        """
        Get all inputs for the team selection process.
        
        Returns:
            Dictionary with all inputs
        """
        if not self.selected_teams or len(self.selected_teams) < 2:
            logger.error("Teams not selected")
            return {}
        
        return {
            'team1': self.selected_teams[0],
            'team2': self.selected_teams[1],
            'constraints': self.constraints,
            'specific_players': self.specific_players,
            'excluded_players': self.excluded_players,
            'favor_spinners': self.favor_spinners,
            'team1_players': self.team1_players
        }
    
    def reset_inputs(self) -> None:
        """
        Reset all inputs to defaults.
        """
        self.selected_teams = []
        self.constraints = {}
        self.specific_players = []
        self.excluded_players = []
        self.favor_spinners = None
        self.team1_players = None
        
        logger.info("Reset all inputs to defaults") 