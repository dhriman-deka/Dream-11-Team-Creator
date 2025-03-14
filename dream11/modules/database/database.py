import pandas as pd
import os
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

from dream11.config import MATCHES_CSV, DELIVERIES_CSV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Team name mapping from historical names to current IPL team names
TEAM_NAME_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kochi Tuskers Kerala': None,  # Team no longer exists
    'Pune Warriors': None,  # Team no longer exists
    'Rising Pune Supergiant': None,  # Team no longer exists
    'Gujarat Lions': None,  # Team no longer exists
    'Rising Pune Supergiants': None,  # Team no longer exists
    'Pune Warriors India': None,  # Team no longer exists
    'Delhi Capitals': 'Delhi Capitals',
    'Mumbai Indians': 'Mumbai Indians',
    'Chennai Super Kings': 'Chennai Super Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
    'Punjab Kings': 'Punjab Kings',
    'Lucknow Super Giants': 'Lucknow Super Giants',
    'Gujarat Titans': 'Gujarat Titans'
}

class Database:
    """
    Database class to handle data storage and retrieval for Dream 11 Team Creator.
    This class is responsible for:
    1. Loading and preprocessing the data
    2. Calculating player statistics
    3. Providing data to other modules
    """
    
    def __init__(self):
        """Initialize the Database object and load the datasets."""
        self.matches_df = None
        self.deliveries_df = None
        self.player_stats = {}
        self.teams = set()
        self.players = set()
        self.player_roles = {}
        self.player_credits = {}
        self.player_teams = {}
        
        # Player pool for selection
        self.player_pool = []
        
        # Load the data
        self._load_data()
        
    def _map_team_names(self, df: pd.DataFrame, team_columns: List[str]) -> pd.DataFrame:
        """
        Map historical team names to current IPL team names in a dataframe.
        
        Args:
            df: DataFrame containing team names
            team_columns: List of column names containing team names
            
        Returns:
            DataFrame with mapped team names
        """
        df = df.copy()
        for col in team_columns:
            if col in df.columns:
                df[col] = df[col].map(lambda x: TEAM_NAME_MAPPING.get(x, x))
        return df

    def _load_data(self):
        """Load the matches and deliveries datasets."""
        try:
            logger.info(f"Loading matches data from {MATCHES_CSV}")
            self.matches_df = pd.read_csv(MATCHES_CSV)
            
            logger.info(f"Loading deliveries data from {DELIVERIES_CSV}")
            self.deliveries_df = pd.read_csv(DELIVERIES_CSV)
            
            # Map team names in both dataframes
            self.matches_df = self._map_team_names(self.matches_df, ['team1', 'team2', 'toss_winner', 'winner'])
            self.deliveries_df = self._map_team_names(self.deliveries_df, ['batting_team', 'bowling_team'])
            
            # Extract teams and players
            self._extract_teams_and_players()
            
            logger.info(f"Data loaded successfully: {len(self.matches_df)} matches, {len(self.players)} players")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _extract_teams_and_players(self):
        """Extract unique teams and players from the datasets."""
        # Extract teams and map to current names
        teams1 = self.matches_df['team1'].unique()
        teams2 = self.matches_df['team2'].unique()
        
        # Map team names to current IPL teams
        mapped_teams = set()
        for team in teams1:
            mapped_team = TEAM_NAME_MAPPING.get(team, team)
            if mapped_team:  # Only add if team still exists
                mapped_teams.add(mapped_team)
        
        for team in teams2:
            mapped_team = TEAM_NAME_MAPPING.get(team, team)
            if mapped_team:  # Only add if team still exists
                mapped_teams.add(mapped_team)
        
        self.teams = mapped_teams
        
        # Extract players
        batters = self.deliveries_df['batter'].unique()
        bowlers = self.deliveries_df['bowler'].unique()
        non_strikers = self.deliveries_df['non_striker'].unique()
        fielders = self.deliveries_df['fielder'].dropna().unique()
        
        self.players.update(batters)
        self.players.update(bowlers)
        self.players.update(non_strikers)
        self.players.update(fielders)
        
        # Remove 'NA' or NaN values
        if 'NA' in self.players:
            self.players.remove('NA')
        
        logger.info(f"Extracted {len(self.teams)} teams and {len(self.players)} players")
    
    def compile_scorecard(self, match_id: int) -> Dict[str, pd.DataFrame]:
        """
        Compile the scorecard for a specific match.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Dictionary containing batting and bowling scorecards
        """
        match_deliveries = self.deliveries_df[self.deliveries_df['match_id'] == match_id]
        
        if match_deliveries.empty:
            logger.warning(f"No deliveries found for match ID {match_id}")
            return {}
        
        # Get match details
        match_details = self.matches_df[self.matches_df['id'] == match_id].iloc[0]
        team1 = match_details['team1']
        team2 = match_details['team2']
        
        # Skip matches with teams that no longer exist
        if not team1 or not team2:
            logger.warning(f"Skipping match {match_id} as one or both teams no longer exist")
            return {}
        
        # Compile batting scorecard
        batting_stats = []
        
        for team in [team1, team2]:
            team_deliveries = match_deliveries[match_deliveries['batting_team'] == team]
            
            if team_deliveries.empty:
                continue
            
            # Group by batsman
            batsmen = team_deliveries['batter'].unique()
            
            for batsman in batsmen:
                batsman_deliveries = team_deliveries[team_deliveries['batter'] == batsman]
                
                runs = batsman_deliveries['batsman_runs'].sum()
                balls_faced = len(batsman_deliveries)
                fours = len(batsman_deliveries[batsman_deliveries['batsman_runs'] == 4])
                sixes = len(batsman_deliveries[batsman_deliveries['batsman_runs'] == 6])
                
                # Check if batsman was dismissed
                dismissed = False
                dismissal_kind = None
                
                if not batsman_deliveries[batsman_deliveries['player_dismissed'] == batsman].empty:
                    dismissed = True
                    dismissal_kind = batsman_deliveries[batsman_deliveries['player_dismissed'] == batsman]['dismissal_kind'].iloc[0]
                
                # Calculate strike rate
                strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
                
                batting_stats.append({
                    'match_id': match_id,
                    'team': team,
                    'player': batsman,
                    'runs': runs,
                    'balls_faced': balls_faced,
                    'fours': fours,
                    'sixes': sixes,
                    'strike_rate': strike_rate,
                    'dismissed': dismissed,
                    'dismissal_kind': dismissal_kind
                })
        
        # Compile bowling scorecard
        bowling_stats = []
        
        for team in [team1, team2]:
            # For bowling stats, we need to find the overs where the team is bowling
            team_deliveries = match_deliveries[match_deliveries['bowling_team'] == team]
            
            if team_deliveries.empty:
                continue
            
            # Group by bowler
            bowlers = team_deliveries['bowler'].unique()
            
            for bowler in bowlers:
                bowler_deliveries = team_deliveries[team_deliveries['bowler'] == bowler]
                
                overs_full = len(bowler_deliveries['over'].unique())
                # Check if any over is incomplete (< 6 legal deliveries)
                for over in bowler_deliveries['over'].unique():
                    over_deliveries = bowler_deliveries[bowler_deliveries['over'] == over]
                    legal_deliveries = len(over_deliveries[~over_deliveries['extras_type'].isin(['wides', 'noballs'])])
                    if legal_deliveries < 6:
                        overs_full -= 1
                        overs_partial = legal_deliveries / 6
                        overs = overs_full + overs_partial
                    else:
                        overs = overs_full
                
                runs_conceded = bowler_deliveries['total_runs'].sum()
                
                # Count wickets
                wickets = len(bowler_deliveries[bowler_deliveries['is_wicket'] == 1])
                
                # Calculate economy rate
                economy_rate = runs_conceded / overs if overs > 0 else 0
                
                # Check for maiden overs
                maidens = 0
                for over in bowler_deliveries['over'].unique():
                    over_deliveries = bowler_deliveries[bowler_deliveries['over'] == over]
                    if over_deliveries['total_runs'].sum() == 0 and len(over_deliveries) >= 6:
                        maidens += 1
                
                bowling_stats.append({
                    'match_id': match_id,
                    'team': team,
                    'player': bowler,
                    'overs': overs,
                    'runs_conceded': runs_conceded,
                    'wickets': wickets,
                    'economy_rate': economy_rate,
                    'maidens': maidens
                })
        
        # Fielding stats
        fielding_stats = []
        
        for team in [team1, team2]:
            # Team is fielding when they are bowling
            team_deliveries = match_deliveries[match_deliveries['bowling_team'] == team]
            
            if team_deliveries.empty:
                continue
            
            # Extract dismissals that involve fielders
            dismissals = team_deliveries[team_deliveries['is_wicket'] == 1]
            
            if dismissals.empty:
                continue
            
            # Count catches, stumpings, and run-outs for each fielder
            fielders = dismissals[dismissals['fielder'].notna()]['fielder'].unique()
            
            for fielder in fielders:
                fielder_dismissals = dismissals[dismissals['fielder'] == fielder]
                
                catches = len(fielder_dismissals[fielder_dismissals['dismissal_kind'] == 'caught'])
                stumpings = len(fielder_dismissals[fielder_dismissals['dismissal_kind'] == 'stumped'])
                run_outs = len(fielder_dismissals[fielder_dismissals['dismissal_kind'] == 'run out'])
                
                fielding_stats.append({
                    'match_id': match_id,
                    'team': team,
                    'player': fielder,
                    'catches': catches,
                    'stumpings': stumpings,
                    'run_outs': run_outs
                })
        
        return {
            'batting': pd.DataFrame(batting_stats) if batting_stats else pd.DataFrame(),
            'bowling': pd.DataFrame(bowling_stats) if bowling_stats else pd.DataFrame(),
            'fielding': pd.DataFrame(fielding_stats) if fielding_stats else pd.DataFrame()
        }
    
    def compile_all_scorecards(self) -> Dict[str, pd.DataFrame]:
        """
        Compile scorecards for all matches and return a dictionary containing batting, bowling, and fielding data.
        
        Returns:
            Dictionary containing DataFrames for batting, bowling, and fielding statistics
        """
        # Check if we already have compiled data
        if hasattr(self, '_compiled_scorecards'):
            logger.info("Using cached compiled scorecards")
            return self._compiled_scorecards
        
        logger.info("Compiling scorecards for all matches...")
        
        # Initialize lists to store statistics
        all_batting_stats = []
        all_bowling_stats = []
        all_fielding_stats = []
        
        # Process matches in chunks of 100
        chunk_size = 100
        total_matches = len(self.matches_df)
        
        for start_idx in range(0, total_matches, chunk_size):
            end_idx = min(start_idx + chunk_size, total_matches)
            chunk_matches = self.matches_df.iloc[start_idx:end_idx]
            
            # Process each match in the chunk
            for _, match in chunk_matches.iterrows():
                match_id = match['id']
                scorecard = self.compile_scorecard(match_id)
                
                if scorecard:
                    if 'batting' in scorecard:
                        all_batting_stats.append(scorecard['batting'])
                    if 'bowling' in scorecard:
                        all_bowling_stats.append(scorecard['bowling'])
                    if 'fielding' in scorecard:
                        all_fielding_stats.append(scorecard['fielding'])
            
            # Log progress
            logger.info(f"Processed {end_idx}/{total_matches} matches")
            
            # Clear memory by concatenating and storing results after each chunk
            if all_batting_stats:
                batting_df = pd.concat(all_batting_stats, ignore_index=True)
                all_batting_stats = [batting_df]
            
            if all_bowling_stats:
                bowling_df = pd.concat(all_bowling_stats, ignore_index=True)
                all_bowling_stats = [bowling_df]
            
            if all_fielding_stats:
                fielding_df = pd.concat(all_fielding_stats, ignore_index=True)
                all_fielding_stats = [fielding_df]
        
        # Create final DataFrames
        result = {
            'batting': all_batting_stats[0] if all_batting_stats else pd.DataFrame(),
            'bowling': all_bowling_stats[0] if all_bowling_stats else pd.DataFrame(),
            'fielding': all_fielding_stats[0] if all_fielding_stats else pd.DataFrame()
        }
        
        # Cache the result
        self._compiled_scorecards = result
        
        logger.info("Compilation completed")
        return result
    
    def assign_player_roles(self, batting_stats: pd.DataFrame, bowling_stats: pd.DataFrame) -> Dict[str, str]:
        """
        Assign roles to players (batsman, bowler, all-rounder, wicket-keeper).
        
        Args:
            batting_stats: DataFrame containing batting statistics
            bowling_stats: DataFrame containing bowling statistics
            
        Returns:
            Dictionary mapping players to their roles
        """
        roles = {}
        
        # Collect all players
        all_players = set(batting_stats['player'].unique()) | set(bowling_stats['player'].unique())
        
        # Group data by player
        batting_by_player = batting_stats.groupby('player')
        bowling_by_player = bowling_stats.groupby('player')
        
        for player in all_players:
            is_batsman = False
            is_bowler = False
            
            # Check if the player is a batsman
            if player in batting_by_player.groups:
                player_batting = batting_by_player.get_group(player)
                total_runs = player_batting['runs'].sum()
                total_balls = player_batting['balls_faced'].sum()
                
                # If the player has scored a significant number of runs, consider them a batsman
                if total_runs > 100 or total_balls > 100:
                    is_batsman = True
            
            # Check if the player is a bowler
            if player in bowling_by_player.groups:
                player_bowling = bowling_by_player.get_group(player)
                total_overs = player_bowling['overs'].sum()
                total_wickets = player_bowling['wickets'].sum()
                
                # If the player has bowled a significant number of overs, consider them a bowler
                if total_overs > 10 or total_wickets > 5:
                    is_bowler = True
            
            # Assign role based on criteria
            if is_batsman and is_bowler:
                roles[player] = 'all-rounder'
            elif is_batsman:
                # For now, we'll assume any player who is primarily a batsman is not a wicket-keeper
                # This will be updated later when we have wicket-keeper data
                roles[player] = 'batsman'
            elif is_bowler:
                roles[player] = 'bowler'
            else:
                # If the player doesn't have enough data, default to batsman
                roles[player] = 'batsman'
        
        # Identify wicket-keepers (this is a simplification since we don't have explicit data)
        # We'll assume players who have a high number of stumpings are wicket-keepers
        fielding_stats = pd.DataFrame()
        if 'fielding' in self.compile_all_scorecards():
            fielding_stats = self.compile_all_scorecards()['fielding']
            if not fielding_stats.empty:
                fielding_by_player = fielding_stats.groupby('player')
                
                for player in all_players:
                    if player in fielding_by_player.groups:
                        player_fielding = fielding_by_player.get_group(player)
                        total_stumpings = player_fielding['stumpings'].sum() if 'stumpings' in player_fielding.columns else 0
                        
                        # If the player has made stumpings, they are likely a wicket-keeper
                        if total_stumpings > 2:
                            roles[player] = 'wicket-keeper'
        
        logger.info(f"Assigned roles to {len(roles)} players")
        return roles
    
    def calculate_player_credits(self, batting_stats: pd.DataFrame, bowling_stats: pd.DataFrame, 
                                fielding_stats: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate credit points for each player based on their performance.
        
        Args:
            batting_stats: DataFrame containing batting statistics
            bowling_stats: DataFrame containing bowling statistics
            fielding_stats: DataFrame containing fielding statistics
            
        Returns:
            Dictionary mapping players to their credit values
        """
        credits = {}
        
        # Collect all players
        all_players = set(batting_stats['player'].unique()) | set(bowling_stats['player'].unique())
        if not fielding_stats.empty:
            all_players |= set(fielding_stats['player'].unique())
        
        # Group data by player
        batting_by_player = batting_stats.groupby('player') if not batting_stats.empty else None
        bowling_by_player = bowling_stats.groupby('player') if not bowling_stats.empty else None
        fielding_by_player = fielding_stats.groupby('player') if not fielding_stats.empty else None
        
        for player in all_players:
            player_credits = 8  # Base credit value
            
            # Add credits based on batting performance
            if batting_by_player is not None and player in batting_by_player.groups:
                player_batting = batting_by_player.get_group(player)
                total_runs = player_batting['runs'].sum()
                total_balls = player_batting['balls_faced'].sum()
                
                # Add credits based on total runs
                player_credits += min(total_runs // 100, 3)
                
                # Add credits based on strike rate
                if total_balls > 50:
                    avg_strike_rate = total_runs / total_balls * 100
                    if avg_strike_rate > 150:
                        player_credits += 2
                    elif avg_strike_rate > 130:
                        player_credits += 1
            
            # Add credits based on bowling performance
            if bowling_by_player is not None and player in bowling_by_player.groups:
                player_bowling = bowling_by_player.get_group(player)
                total_wickets = player_bowling['wickets'].sum()
                total_overs = player_bowling['overs'].sum()
                
                # Add credits based on total wickets
                player_credits += min(total_wickets // 10, 3)
                
                # Add credits based on economy rate
                if total_overs > 20:
                    total_runs = player_bowling['runs_conceded'].sum()
                    economy_rate = total_runs / total_overs
                    if economy_rate < 6:
                        player_credits += 2
                    elif economy_rate < 7.5:
                        player_credits += 1
            
            # Add credits based on fielding performance
            if fielding_by_player is not None and player in fielding_by_player.groups:
                player_fielding = fielding_by_player.get_group(player)
                total_catches = player_fielding['catches'].sum() if 'catches' in player_fielding.columns else 0
                total_stumpings = player_fielding['stumpings'].sum() if 'stumpings' in player_fielding.columns else 0
                
                # Add credits for exceptional fielding
                if total_catches > 15 or total_stumpings > 10:
                    player_credits += 1
            
            # Extra credits for renowned players (this is just a placeholder)
            # In a real implementation, this might be based on ICC rankings or other external data
            if player in ['MS Dhoni', 'V Kohli', 'RG Sharma', 'JJ Bumrah', 'R Ashwin']:
                player_credits += 2
            
            # Cap the credits at 12
            credits[player] = min(player_credits, 12)
        
        logger.info(f"Calculated credits for {len(credits)} players")
        return credits
    
    def create_player_pool(self, team1: str, team2: str) -> List[Dict]:
        """
        Create a pool of players for selection from two teams.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            
        Returns:
            List of player dictionaries with their details
        """
        # Check if the teams exist in our dataset
        if team1 not in self.teams or team2 not in self.teams:
            logger.error(f"One or both teams not found: {team1}, {team2}")
            return []
        
        # Compile all scorecards if not already done
        all_scorecards = self.compile_all_scorecards()
        batting_stats = all_scorecards.get('batting', pd.DataFrame())
        bowling_stats = all_scorecards.get('bowling', pd.DataFrame())
        fielding_stats = all_scorecards.get('fielding', pd.DataFrame())
        
        # Assign roles if not already done
        if not self.player_roles:
            self.player_roles = self.assign_player_roles(batting_stats, bowling_stats)
        
        # Calculate credits if not already done
        if not self.player_credits:
            self.player_credits = self.calculate_player_credits(batting_stats, bowling_stats, fielding_stats)
        
        # Find all players who have played for these teams
        team1_players = set(batting_stats[batting_stats['team'] == team1]['player'].unique())
        team1_players |= set(bowling_stats[bowling_stats['team'] == team1]['player'].unique())
        
        team2_players = set(batting_stats[batting_stats['team'] == team2]['player'].unique())
        team2_players |= set(bowling_stats[bowling_stats['team'] == team2]['player'].unique())
        
        # Create player pool
        player_pool = []
        
        for player in team1_players:
            if player in self.player_roles and player in self.player_credits:
                player_data = {
                    'name': player,
                    'team': team1,
                    'role': self.player_roles[player],
                    'credits': self.player_credits[player],
                    'is_spinner': self._is_spinner(player, bowling_stats)
                }
                player_pool.append(player_data)
        
        for player in team2_players:
            if player in self.player_roles and player in self.player_credits:
                player_data = {
                    'name': player,
                    'team': team2,
                    'role': self.player_roles[player],
                    'credits': self.player_credits[player],
                    'is_spinner': self._is_spinner(player, bowling_stats)
                }
                player_pool.append(player_data)
        
        # Store the player pool
        self.player_pool = player_pool
        logger.info(f"Created player pool with {len(player_pool)} players from {team1} and {team2}")
        
        return player_pool
    
    def _is_spinner(self, player: str, bowling_stats: pd.DataFrame) -> bool:
        """
        Determine if a bowler is a spinner based on their bowling statistics.
        This is a simplified approach since we don't have explicit data on bowling style.
        
        Args:
            player: Name of the player
            bowling_stats: DataFrame containing bowling statistics
            
        Returns:
            Boolean indicating whether the player is likely a spinner
        """
        # This is a simplified approach - in reality, you would need more data
        # We'll use a list of known spinners for demonstration
        known_spinners = [
            'R Ashwin', 'Harbhajan Singh', 'A Kumble', 'Rashid Khan', 
            'Yuzvendra Chahal', 'Imran Tahir', 'Ravindra Jadeja', 
            'Sunil Narine', 'Piyush Chawla', 'Amit Mishra'
        ]
        
        return player in known_spinners
    
    def get_player_performance(self, player: str, recent_matches: int = 10) -> Dict:
        """
        Get performance statistics for a player from recent matches.
        
        Args:
            player: Name of the player
            recent_matches: Number of recent matches to consider
            
        Returns:
            Dictionary containing performance statistics
        """
        all_scorecards = self.compile_all_scorecards()
        batting_stats = all_scorecards.get('batting', pd.DataFrame())
        bowling_stats = all_scorecards.get('bowling', pd.DataFrame())
        fielding_stats = all_scorecards.get('fielding', pd.DataFrame())
        
        # Filter for the specific player
        player_batting = batting_stats[batting_stats['player'] == player] if not batting_stats.empty else pd.DataFrame()
        player_bowling = bowling_stats[bowling_stats['player'] == player] if not bowling_stats.empty else pd.DataFrame()
        player_fielding = fielding_stats[fielding_stats['player'] == player] if not fielding_stats.empty else pd.DataFrame()
        
        # Sort by match_id (assuming higher match_id means more recent match)
        if not player_batting.empty:
            player_batting = player_batting.sort_values('match_id', ascending=False)
        if not player_bowling.empty:
            player_bowling = player_bowling.sort_values('match_id', ascending=False)
        if not player_fielding.empty:
            player_fielding = player_fielding.sort_values('match_id', ascending=False)
        
        # Take only the recent matches
        recent_batting = player_batting.head(recent_matches) if not player_batting.empty else pd.DataFrame()
        recent_bowling = player_bowling.head(recent_matches) if not player_bowling.empty else pd.DataFrame()
        recent_fielding = player_fielding.head(recent_matches) if not player_fielding.empty else pd.DataFrame()
        
        # Calculate performance metrics
        performance = {
            'batting': {
                'matches_played': len(recent_batting),
                'total_runs': recent_batting['runs'].sum() if not recent_batting.empty else 0,
                'avg_runs': recent_batting['runs'].mean() if not recent_batting.empty else 0,
                'total_balls_faced': recent_batting['balls_faced'].sum() if not recent_batting.empty else 0,
                'strike_rate': (recent_batting['runs'].sum() / recent_batting['balls_faced'].sum() * 100 
                                if not recent_batting.empty and recent_batting['balls_faced'].sum() > 0 else 0),
                'fifties': len(recent_batting[recent_batting['runs'] >= 50]) if not recent_batting.empty else 0,
                'hundreds': len(recent_batting[recent_batting['runs'] >= 100]) if not recent_batting.empty else 0,
                'fours': recent_batting['fours'].sum() if not recent_batting.empty and 'fours' in recent_batting.columns else 0,
                'sixes': recent_batting['sixes'].sum() if not recent_batting.empty and 'sixes' in recent_batting.columns else 0
            },
            'bowling': {
                'matches_played': len(recent_bowling),
                'total_overs': recent_bowling['overs'].sum() if not recent_bowling.empty else 0,
                'total_runs_conceded': recent_bowling['runs_conceded'].sum() if not recent_bowling.empty else 0,
                'total_wickets': recent_bowling['wickets'].sum() if not recent_bowling.empty else 0,
                'avg_economy': (recent_bowling['runs_conceded'].sum() / recent_bowling['overs'].sum() 
                               if not recent_bowling.empty and recent_bowling['overs'].sum() > 0 else 0),
                'bowling_avg': (recent_bowling['runs_conceded'].sum() / recent_bowling['wickets'].sum() 
                               if not recent_bowling.empty and recent_bowling['wickets'].sum() > 0 else 0),
                'maidens': recent_bowling['maidens'].sum() if not recent_bowling.empty and 'maidens' in recent_bowling.columns else 0
            },
            'fielding': {
                'matches_played': len(recent_fielding),
                'catches': recent_fielding['catches'].sum() if not recent_fielding.empty and 'catches' in recent_fielding.columns else 0,
                'stumpings': recent_fielding['stumpings'].sum() if not recent_fielding.empty and 'stumpings' in recent_fielding.columns else 0,
                'run_outs': recent_fielding['run_outs'].sum() if not recent_fielding.empty and 'run_outs' in recent_fielding.columns else 0,
                'fielding_score': (
                    (recent_fielding['catches'].sum() if not recent_fielding.empty and 'catches' in recent_fielding.columns else 0) +
                    (recent_fielding['stumpings'].sum() * 1.5 if not recent_fielding.empty and 'stumpings' in recent_fielding.columns else 0) +
                    (recent_fielding['run_outs'].sum() * 1.5 if not recent_fielding.empty and 'run_outs' in recent_fielding.columns else 0)
                )
            }
        }
        
        # Calculate player consistency as described in the paper
        batting_consistency = performance['batting']['avg_runs'] > 25
        bowling_consistency = (performance['bowling']['avg_economy'] < 7.0 or 
                               performance['bowling']['total_wickets'] / max(1, performance['bowling']['matches_played']) > 1)
        
        # Determine role based on available data
        if player in self.player_roles:
            role = self.player_roles[player]
        else:
            if performance['batting']['matches_played'] > 0 and performance['bowling']['matches_played'] > 0:
                role = 'all-rounder'
            elif performance['batting']['matches_played'] > 0:
                role = 'batsman'
            elif performance['bowling']['matches_played'] > 0:
                role = 'bowler'
            else:
                role = 'unknown'
        
        # Set consistency based on role
        if role == 'batsman':
            performance['consistency'] = batting_consistency
        elif role == 'bowler':
            performance['consistency'] = bowling_consistency
        elif role == 'all-rounder':
            performance['consistency'] = (performance['batting']['avg_runs'] > 15 and 
                                         (performance['bowling']['avg_economy'] < 7.0 or 
                                          performance['bowling']['total_wickets'] / max(1, performance['bowling']['matches_played']) > 1))
        elif role == 'wicket-keeper':
            performance['consistency'] = (performance['batting']['avg_runs'] > 25 and 
                                         performance['fielding']['fielding_score'] > 2)
        else:
            performance['consistency'] = False
        
        return performance
    
    def get_match_venue_details(self, match_id: int) -> Dict:
        """
        Get venue details for a specific match.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Dictionary containing venue details
        """
        match_details = self.matches_df[self.matches_df['id'] == match_id]
        
        if match_details.empty:
            logger.warning(f"No match found with ID {match_id}")
            return {}
        
        match_details = match_details.iloc[0]
        
        venue_details = {
            'venue': match_details['venue'],
            'city': match_details['city'] if 'city' in match_details else None,
            'date': match_details['date']
        }
        
        return venue_details
        
    def get_player_pool(self) -> List[Dict]:
        """
        Get the current player pool.
        
        Returns:
            List of player dictionaries
        """
        return self.player_pool 