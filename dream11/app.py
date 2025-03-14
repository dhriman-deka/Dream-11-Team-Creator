from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import logging
import os
import pickle
from datetime import datetime
import json
import numpy as np

from dream11.modules.database import Database
from dream11.modules.ml import MLModel
from dream11.modules.input import InputHandler
from dream11.modules.selection import TeamSelector
from dream11.config import DEFAULT_CONSTRAINTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  # For session management
app.json_encoder = NumpyJSONEncoder  # Set custom JSON encoder

# Initialize models directory
os.makedirs('models', exist_ok=True)

# Global variables for modules
db = None
ml_model = None
input_handler = None
team_selector = None


def initialize_modules():
    """Initialize all modules."""
    global db, ml_model, input_handler, team_selector
    
    logger.info("Initializing modules...")
    
    try:
        # Initialize database
        db = Database()
        
        # Initialize ML model
        ml_model = MLModel(db)
        
        # Try to load a pre-trained model if available
        model_path = os.path.join('models', 'dream11_model.pkl')
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            ml_model.load_model(model_path)
        else:
            logger.info("No pre-trained model found. A new model will be trained when needed.")
        
        # Initialize input handler and team selector
        input_handler = InputHandler(db)
        team_selector = TeamSelector(db, ml_model)
        
        # Generate evaluation metrics if they don't exist yet
        if ml_model.model is not None and not os.path.exists('evaluation_metrics.txt'):
            logger.info("Generating evaluation metrics report...")
            ml_model.generate_evaluation_report()
        
        logger.info("Modules initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing modules: {str(e)}")
        return False


# Initialize modules when the app starts
with app.app_context():
    if not initialize_modules():
        logger.error("Failed to initialize application modules")
        raise RuntimeError("Failed to initialize application modules")


@app.route('/')
def index():
    """Home page."""
    if not all([db, ml_model, input_handler, team_selector]):
        if not initialize_modules():
            return render_template('error.html', message="Failed to initialize application. Please try again later.")
    return render_template('index.html')


@app.route('/teams', methods=['GET', 'POST'])
def teams():
    """Page for selecting teams."""
    if not all([db, ml_model, input_handler, team_selector]):
        if not initialize_modules():
            return render_template('error.html', message="Failed to initialize application. Please try again later.")
    
    if request.method == 'POST':
        team1 = request.form.get('team1')
        team2 = request.form.get('team2')
        
        if team1 and team2:
            success = input_handler.select_teams(team1, team2)
            if success:
                session['team1'] = team1
                session['team2'] = team2
                return redirect(url_for('constraints'))
    
    # Get available teams
    available_teams = sorted(input_handler.get_available_teams())
    return render_template('teams.html', teams=available_teams)


@app.route('/constraints', methods=['GET', 'POST'])
def constraints():
    """Page for setting constraints."""
    if 'team1' not in session or 'team2' not in session:
        return redirect(url_for('teams'))
    
    if request.method == 'POST':
        # Parse constraints from form
        constraints = {}
        
        # Total credits
        total_credits = request.form.get('total_credits')
        if total_credits:
            constraints['TOTAL_CREDITS'] = int(total_credits)
        
        # Batsmen
        min_batsmen = request.form.get('min_batsmen')
        max_batsmen = request.form.get('max_batsmen')
        if min_batsmen and max_batsmen:
            constraints['BATSMEN'] = {
                'MIN': int(min_batsmen),
                'MAX': int(max_batsmen)
            }
        
        # Bowlers
        min_bowlers = request.form.get('min_bowlers')
        max_bowlers = request.form.get('max_bowlers')
        if min_bowlers and max_bowlers:
            constraints['BOWLERS'] = {
                'MIN': int(min_bowlers),
                'MAX': int(max_bowlers)
            }
        
        # All-rounders
        min_all_rounders = request.form.get('min_all_rounders')
        max_all_rounders = request.form.get('max_all_rounders')
        if min_all_rounders and max_all_rounders:
            constraints['ALL_ROUNDERS'] = {
                'MIN': int(min_all_rounders),
                'MAX': int(max_all_rounders)
            }
        
        # Wicket-keepers
        min_wicket_keepers = request.form.get('min_wicket_keepers')
        max_wicket_keepers = request.form.get('max_wicket_keepers')
        if min_wicket_keepers and max_wicket_keepers:
            constraints['WICKET_KEEPERS'] = {
                'MIN': int(min_wicket_keepers),
                'MAX': int(max_wicket_keepers)
            }
        
        # Max players from team
        max_players = request.form.get('max_players_from_team')
        if max_players:
            constraints['MAX_PLAYERS_FROM_TEAM'] = int(max_players)
        
        # Store constraints in session and update team selector
        session['constraints'] = constraints
        team_selector.update_constraints(constraints)
        
        # Specific players
        specific_players = request.form.get('specific_players', '')
        if specific_players:
            specific_players_list = [p.strip() for p in specific_players.split(',') if p.strip()]
            input_handler.set_specific_players(specific_players_list)
            session['specific_players'] = specific_players_list
        
        # Excluded players
        excluded_players = request.form.get('excluded_players', '')
        if excluded_players:
            excluded_players_list = [p.strip() for p in excluded_players.split(',') if p.strip()]
            input_handler.set_excluded_players(excluded_players_list)
            session['excluded_players'] = excluded_players_list
        
        # Spinner preference
        favor_spinners = request.form.get('favor_spinners')
        if favor_spinners:
            favor_spinners_bool = (favor_spinners == 'yes')
            input_handler.set_spinner_preference(favor_spinners_bool)
            session['favor_spinners'] = favor_spinners_bool
        
        # Team distribution
        team1_players = request.form.get('team1_players')
        if team1_players and team1_players.isdigit():
            input_handler.set_team_distribution(int(team1_players))
            session['team1_players'] = int(team1_players)
        
        return redirect(url_for('select_team'))
    
    # Get default constraints
    default_constraints = DEFAULT_CONSTRAINTS
    team1 = session.get('team1')
    team2 = session.get('team2')
    
    return render_template('constraints.html', constraints=default_constraints, team1=team1, team2=team2)


@app.route('/select_team')
def select_team():
    """Generate and display the selected team."""
    if not all([db, ml_model, input_handler, team_selector]):
        return render_template('error.html', message="Application not properly initialized")
    
    # Get selected teams from session
    team1 = session.get('team1')
    team2 = session.get('team2')
    
    if not team1 or not team2:
        return redirect(url_for('teams'))
    
    # Generate team
    selected_team = team_selector.select_team(team1, team2)
    
    if not selected_team:
        return render_template('error.html', message="Failed to generate team")
    
    # Get captain and vice-captain suggestions
    captain, vice_captain = team_selector.suggest_captain_vice_captain()
    
    # Get team summary
    team_summary = team_selector.get_team_summary()
    
    # Calculate role counts and team counts
    role_counts = {
        'batsman': 0,
        'bowler': 0,
        'all_rounder': 0,
        'wicket_keeper': 0
    }
    
    team_counts = {
        team1: 0,
        team2: 0
    }
    
    # Convert any NumPy types to standard Python types for JSON serialization
    for i, player in enumerate(selected_team):
        # Convert player data to standard types
        selected_team[i] = {k: float(v) if isinstance(v, np.floating) else 
                             int(v) if isinstance(v, np.integer) else v 
                             for k, v in player.items()}
        
        # Update role counts safely
        role = player.get('role', '').lower()
        if role in role_counts:
            role_counts[role] += 1
        
        # Update team counts
        team_counts[player['team']] += 1
    
    # Store selected team in session
    session['selected_team'] = selected_team
    session['captain'] = captain
    session['vice_captain'] = vice_captain
    
    return render_template(
        'result.html',
        team1=team1,
        team2=team2,
        selected_team=selected_team,
        captain=captain,
        vice_captain=vice_captain,
        role_counts=role_counts,
        team_counts=team_counts,
        total_credits=sum(player['credits'] for player in selected_team)
    )


@app.route('/export_team')
def export_team():
    """Export the selected team to a JSON file."""
    if 'selected_team' not in session:
        return redirect(url_for('teams'))
    
    selected_team = session['selected_team']
    captain = session.get('captain')
    vice_captain = session.get('vice_captain')
    
    # Create export data
    export_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'team': selected_team,
        'captain': captain,
        'vice_captain': vice_captain,
        'total_credits': sum(player['credits'] for player in selected_team)
    }
    
    # Create exports directory if it doesn't exist
    os.makedirs('exports', exist_ok=True)
    
    # Generate filename with timestamp
    filename = f"dream11_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('exports', filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=4)
    
    return jsonify({
        'success': True,
        'message': f'Team exported successfully to {filename}',
        'filename': filename
    })


@app.route('/player/<player_name>')
def player_details(player_name):
    """Page for showing player details."""
    player_info = input_handler.get_player_details(player_name)
    
    if not player_info:
        return render_template('error.html', message=f"Player '{player_name}' not found")
    
    return render_template('player_details.html', player=player_info)


@app.route('/reset', methods=['POST'])
def reset():
    """Reset all inputs and selections."""
    input_handler.reset_inputs()
    session.clear()
    return redirect(url_for('index'))


@app.route('/api/teams', methods=['GET'])
def api_teams():
    """API endpoint for getting available teams."""
    teams = input_handler.get_available_teams()
    return jsonify({'teams': teams})


@app.route('/api/players', methods=['GET'])
def api_players():
    """API endpoint for getting players from selected teams."""
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    
    if not team1 or not team2:
        return jsonify({'error': 'Both team1 and team2 are required'}), 400
    
    # Create player pool
    player_pool = db.create_player_pool(team1, team2)
    
    return jsonify({'players': player_pool})


@app.route('/api/select_team', methods=['POST'])
def api_select_team():
    """API endpoint for selecting a team."""
    data = request.json
    
    team1 = data.get('team1')
    team2 = data.get('team2')
    constraints = data.get('constraints', {})
    specific_players = data.get('specific_players', [])
    excluded_players = data.get('excluded_players', [])
    favor_spinners = data.get('favor_spinners')
    team1_players = data.get('team1_players')
    
    if not team1 or not team2:
        return jsonify({'error': 'Both team1 and team2 are required'}), 400
    
    # Update constraints
    if constraints:
        team_selector.update_constraints(constraints)
    
    # Select the team
    selected_team = team_selector.select_team(
        team1=team1,
        team2=team2,
        specific_players=specific_players,
        excluded_players=excluded_players,
        favor_spinners=favor_spinners,
        team1_players=team1_players
    )
    
    if not selected_team:
        return jsonify({'error': 'Team selection failed. Please try different constraints.'}), 400
    
    # Get captain and vice-captain
    captain, vice_captain = team_selector.suggest_captain_vice_captain()
    
    # Get team summary
    team_summary = team_selector.get_team_summary()
    
    return jsonify({
        'team': selected_team,
        'captain': captain,
        'vice_captain': vice_captain,
        'summary': team_summary
    })


def main():
    """Main entry point for the application."""
    initialize_modules()
    app.run(debug=True, port=5001)


if __name__ == '__main__':
    main()
