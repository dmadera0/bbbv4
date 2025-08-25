import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLBGamePredictor:
    """MLB game outcome and run prediction model with daily retraining"""
    
    def __init__(self):
        self.winner_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.runs_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.team_stats = {}
        self.season_data = pd.DataFrame()
        
    def fetch_game_data(self, date):
        """
        Fetch MLB game data for a specific date
        Note: In production, replace with actual MLB API calls
        """
        # Simulated data structure - replace with actual API calls
        # You would use MLB StatsAPI or similar service here
        games_data = {
            'date': date,
            'home_team': ['NYY', 'LAD', 'HOU'],
            'away_team': ['BOS', 'SF', 'OAK'],
            'home_runs': [5, 7, 3],
            'away_runs': [3, 6, 4],
            'home_hits': [10, 12, 7],
            'away_hits': [8, 11, 9],
            'home_errors': [1, 0, 2],
            'away_errors': [2, 1, 1]
        }
        return pd.DataFrame(games_data)
    
    def calculate_team_features(self, team, date, last_n_games=10):
        """Calculate rolling statistics for a team"""
        team_games = self.season_data[
            ((self.season_data['home_team'] == team) | 
             (self.season_data['away_team'] == team)) &
            (self.season_data['date'] < date)
        ].tail(last_n_games)
        
        if len(team_games) == 0:
            return {
                'avg_runs_scored': 4.5,  # League average defaults
                'avg_runs_allowed': 4.5,
                'win_pct': 0.500,
                'recent_form': 0.5,
                'avg_hits': 9,
                'avg_errors': 1
            }
        
        features = {}
        home_games = team_games[team_games['home_team'] == team]
        away_games = team_games[team_games['away_team'] == team]
        
        # Calculate offensive stats
        runs_scored = (
            home_games['home_runs'].sum() + 
            away_games['away_runs'].sum()
        )
        features['avg_runs_scored'] = runs_scored / len(team_games)
        
        # Calculate defensive stats
        runs_allowed = (
            home_games['away_runs'].sum() + 
            away_games['home_runs'].sum()
        )
        features['avg_runs_allowed'] = runs_allowed / len(team_games)
        
        # Calculate win percentage
        wins = (
            len(home_games[home_games['home_runs'] > home_games['away_runs']]) +
            len(away_games[away_games['away_runs'] > away_games['home_runs']])
        )
        features['win_pct'] = wins / len(team_games)
        
        # Recent form (last 5 games)
        recent = team_games.tail(5)
        recent_wins = 0
        for _, game in recent.iterrows():
            if game['home_team'] == team:
                if game['home_runs'] > game['away_runs']:
                    recent_wins += 1
            else:
                if game['away_runs'] > game['home_runs']:
                    recent_wins += 1
        features['recent_form'] = recent_wins / min(5, len(recent))
        
        # Average hits and errors
        hits = (
            home_games['home_hits'].sum() + 
            away_games['away_hits'].sum()
        )
        features['avg_hits'] = hits / len(team_games)
        
        errors = (
            home_games['home_errors'].sum() + 
            away_games['away_errors'].sum()
        )
        features['avg_errors'] = errors / len(team_games)
        
        return features
    
    def prepare_game_features(self, home_team, away_team, date):
        """Prepare features for a single game prediction"""
        home_stats = self.calculate_team_features(home_team, date)
        away_stats = self.calculate_team_features(away_team, date)
        
        features = {
            # Home team features
            'home_avg_runs_scored': home_stats['avg_runs_scored'],
            'home_avg_runs_allowed': home_stats['avg_runs_allowed'],
            'home_win_pct': home_stats['win_pct'],
            'home_recent_form': home_stats['recent_form'],
            'home_avg_hits': home_stats['avg_hits'],
            'home_avg_errors': home_stats['avg_errors'],
            
            # Away team features
            'away_avg_runs_scored': away_stats['avg_runs_scored'],
            'away_avg_runs_allowed': away_stats['avg_runs_allowed'],
            'away_win_pct': away_stats['win_pct'],
            'away_recent_form': away_stats['recent_form'],
            'away_avg_hits': away_stats['avg_hits'],
            'away_avg_errors': away_stats['avg_errors'],
            
            # Matchup features
            'run_diff': home_stats['avg_runs_scored'] - away_stats['avg_runs_allowed'],
            'defensive_diff': away_stats['avg_runs_scored'] - home_stats['avg_runs_allowed'],
            'form_diff': home_stats['recent_form'] - away_stats['recent_form'],
            
            # Home field advantage
            'home_advantage': 1
        }
        
        return features
    
    def train_models(self):
        """Train both winner and total runs models"""
        if len(self.season_data) < 20:
            print("Not enough data for training (need at least 20 games)")
            return
        
        # Prepare training data
        X = []
        y_winner = []
        y_runs = []
        
        for _, game in self.season_data.iterrows():
            features = self.prepare_game_features(
                game['home_team'], 
                game['away_team'], 
                game['date']
            )
            X.append(list(features.values()))
            
            # Winner: 1 for home, 0 for away
            y_winner.append(1 if game['home_runs'] > game['away_runs'] else 0)
            
            # Total runs
            y_runs.append(game['home_runs'] + game['away_runs'])
        
        X = np.array(X)
        y_winner = np.array(y_winner)
        y_runs = np.array(y_runs)
        
        # Split data
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X, y_winner, test_size=0.2, random_state=42
        )
        _, _, y_run_train, y_run_test = train_test_split(
            X, y_runs, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.winner_model.fit(X_train_scaled, y_win_train)
        self.runs_model.fit(X_train_scaled, y_run_train)
        
        # Calculate accuracy
        win_accuracy = self.winner_model.score(X_test_scaled, y_win_test)
        print(f"Winner prediction accuracy: {win_accuracy:.2%}")
        
        # Calculate RMSE for runs
        from sklearn.metrics import mean_squared_error
        runs_pred = self.runs_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_run_test, runs_pred))
        print(f"Runs prediction RMSE: {rmse:.2f}")
    
    def predict_game(self, home_team, away_team, date):
        """Predict a single game outcome"""
        features = self.prepare_game_features(home_team, away_team, date)
        X = np.array([list(features.values())])
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except:
            # If scaler not fitted yet, use raw features
            X_scaled = X
        
        # Predict winner probability
        try:
            winner_proba = self.winner_model.predict_proba(X_scaled)[0]
            predicted_winner = home_team if winner_proba[1] > 0.5 else away_team
            confidence = max(winner_proba)
        except:
            # Default prediction if model not trained
            predicted_winner = home_team
            confidence = 0.5
        
        # Predict total runs
        try:
            predicted_runs = self.runs_model.predict(X_scaled)[0]
        except:
            predicted_runs = 9.0  # League average
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'predicted_total_runs': round(predicted_runs, 1),
            'confidence_index': round(confidence, 3)
        }
    
    def predict_and_update(self, date):
        """Predict today's games and update with results"""
        # Get today's games
        todays_games = self.fetch_game_data(date)
        
        predictions = []
        for _, game in todays_games.iterrows():
            # Make prediction
            pred = self.predict_game(
                game['home_team'], 
                game['away_team'], 
                date
            )
            
            # Add actual results
            actual_winner = (
                game['home_team'] if game['home_runs'] > game['away_runs'] 
                else game['away_team']
            )
            actual_runs = game['home_runs'] + game['away_runs']
            
            pred['actual_winner'] = actual_winner
            pred['actual_total_runs'] = actual_runs
            pred['correct_prediction'] = pred['predicted_winner'] == actual_winner
            
            predictions.append(pred)
        
        # Add today's games to season data
        self.season_data = pd.concat([self.season_data, todays_games], ignore_index=True)
        
        # Retrain models with new data
        if len(self.season_data) >= 20:
            print(f"\nRetraining models with {len(self.season_data)} games...")
            self.train_models()
        
        return pd.DataFrame(predictions)
    
    def run_season_simulation(self, start_date, num_days=30):
        """Simulate predictions for a season"""
        current_date = start_date
        all_predictions = []
        
        for day in range(num_days):
            print(f"\n{'='*50}")
            print(f"Date: {current_date}")
            print('='*50)
            
            # Make predictions and update
            daily_predictions = self.predict_and_update(current_date)
            
            # Display results
            for _, pred in daily_predictions.iterrows():
                print(f"\n{pred['home_team']} vs {pred['away_team']}")
                print(f"  Predicted Winner: {pred['predicted_winner']}")
                print(f"  Actual Winner: {pred['actual_winner']}")
                print(f"  Predicted Total Runs: {pred['predicted_total_runs']}")
                print(f"  Actual Total Runs: {pred['actual_total_runs']}")
                print(f"  Confidence Index: {pred['confidence_index']:.3f}")
                print(f"  Result: {'✓ Correct' if pred['correct_prediction'] else '✗ Incorrect'}")
            
            all_predictions.append(daily_predictions)
            
            # Calculate running accuracy
            if len(all_predictions) > 0:
                all_preds_df = pd.concat(all_predictions, ignore_index=True)
                accuracy = all_preds_df['correct_prediction'].mean()
                print(f"\nSeason Accuracy: {accuracy:.2%} ({len(all_preds_df)} games)")
            
            current_date += timedelta(days=1)
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def save_model(self, filepath='mlb_predictor.pkl'):
        """Save the trained model"""
        model_data = {
            'winner_model': self.winner_model,
            'runs_model': self.runs_model,
            'scaler': self.scaler,
            'season_data': self.season_data
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='mlb_predictor.pkl'):
        """Load a previously trained model"""
        model_data = joblib.load(filepath)
        self.winner_model = model_data['winner_model']
        self.runs_model = model_data['runs_model']
        self.scaler = model_data['scaler']
        self.season_data = model_data['season_data']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = MLBGamePredictor()
    
    # Run season simulation
    start_date = datetime(2024, 4, 1)  # MLB season typically starts in April
    season_results = predictor.run_season_simulation(start_date, num_days=7)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SEASON SUMMARY")
    print("="*60)
    print(f"Total Games Predicted: {len(season_results)}")
    print(f"Correct Predictions: {season_results['correct_prediction'].sum()}")
    print(f"Overall Accuracy: {season_results['correct_prediction'].mean():.2%}")
    print(f"Average Confidence: {season_results['confidence_index'].mean():.3f}")
    
    # Save the model
    predictor.save_model()