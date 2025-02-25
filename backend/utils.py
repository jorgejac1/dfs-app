"""
utils.py
---------
This module contains helper functions to:
- Fetch and cache data from nba_api endpoints.
- Clean and validate incoming data.
- Compute advanced DFS features such as effective FG% (eFG%), true shooting attempts (TSA),
  rolling averages, home/away splits, lineup efficiency, and other contextual factors.
All functions include detailed comments and use caching to reduce API calls.
"""

import datetime
import math
import pandas as pd
from cachetools import cached, TTLCache
from concurrent.futures import ThreadPoolExecutor

# nba_api endpoints
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playergamelog,
    leagueDashTeamStats,
    boxscoreusagev3,
    boxscoredefensivev2,
    defensehub,
    boxscoresummaryv2,
    boxscoreadvancedv3,
    playerfantasyprofile,
    playerdashboardbygeneralsplits,
    teamdashlineups,
    playerestimatedmetrics,
    playergamelogs
)

# Set up a TTL cache (cache up to 200 items for 1 hour)
cache = TTLCache(maxsize=200, ttl=3600)
executor = ThreadPoolExecutor(max_workers=10)

def get_current_season():
    """
    Determine the current NBA season string.
    If the current month is before July, assume the season started last year.
    """
    now = datetime.datetime.now()
    year = now.year
    return f"{year-1}-{str(year)[-2:]}" if now.month < 7 else f"{year}-{str(year+1)[-2:]}"

@cached(cache)
def get_nba_player_id(player_name):
    """
    Given a player's full name, return the NBA player ID.
    """
    result = players.find_players_by_full_name(player_name)
    return result[0]['id'] if result else None

@cached(cache)
def get_player_season_stats(player_id, season=None):
    """
    Fetch a player's season statistics using PlayerGameLog.
    Returns a tuple: (dictionary of averages, DataFrame of game logs).
    """
    if season is None:
        season = get_current_season()
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = log.get_data_frames()[0]
        # Convert key columns to numeric values
        for col in ['PTS','REB','AST','STL','BLK','TOV','FGM','FG3M','MIN','FGA','FTA']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        stats = {
            'PTS': df['PTS'].mean(),
            'REB': df['REB'].mean(),
            'AST': df['AST'].mean(),
            'STL': df['STL'].mean(),
            'BLK': df['BLK'].mean(),
            'TOV': df['TOV'].mean(),
            'FGM': df['FGM'].mean(),
            'FG3M': df['FG3M'].mean(),
            'MIN': df['MIN'].mean(),
            'FGA': df['FGA'].mean(),
            'FTA': df['FTA'].mean(),
        }
        return stats, df
    except Exception as e:
        print(f"Error fetching season stats for player {player_id}: {e}")
        return None, None

@cached(cache)
def get_team_pace(team_abbreviation, season=None):
    """
    Retrieve a team's pace from leagueDashTeamStats.
    """
    if season is None:
        season = get_current_season()
    try:
        stats = leagueDashTeamStats.LeagueDashTeamStats(season=season, season_type_all_star='Regular Season')
        df = stats.get_data_frames()[0]
        row = df[df['TEAM_ABBREVIATION'] == team_abbreviation.upper()]
        return row['PACE'].iloc[0] if not row.empty else 100.0
    except Exception as e:
        print(f"Error fetching pace for team {team_abbreviation}: {e}")
        return 100.0

@cached(cache)
def get_league_average_pace(season):
    """
    Compute and return the league average pace.
    """
    try:
        stats = leagueDashTeamStats.LeagueDashTeamStats(season=season, season_type_all_star='Regular Season')
        df = stats.get_data_frames()[0]
        return df['PACE'].mean()
    except Exception as e:
        print(f"Error fetching league average pace: {e}")
        return 100.0

@cached(cache)
def get_league_average_def_rating(season):
    """
    Compute and return the league average defensive rating.
    """
    try:
        stats = leagueDashTeamStats.LeagueDashTeamStats(season=season, season_type_all_star='Regular Season')
        df = stats.get_data_frames()[0]
        return df['DEF_RATING'].mean()
    except Exception as e:
        print(f"Error fetching league average defensive rating: {e}")
        return None

def get_player_usage_boxscore(player_id, recent_game_id):
    """
    Retrieve usage percentage for a player in a given game using BoxScoreUsageV3.
    """
    try:
        usage = boxscoreusagev3.BoxScoreUsageV3(game_id=recent_game_id)
        df = usage.player_stats.get_data_frame()
        data = df[df['personId'] == int(player_id)]
        if data.empty:
            return None, None
        return data['usagePercentage'].iloc[0], df['usagePercentage'].mean()
    except Exception as e:
        print(f"Error fetching usage for player {player_id}: {e}")
        return None, None

def parse_opponent_from_game_info(game_info, player_team):
    """
    Extract the opponent team abbreviation from a matchup string (e.g., "MIA@MIL").
    """
    try:
        matchup = game_info.split()[0]
        teams = matchup.split('@')
        if len(teams) != 2:
            return None
        return teams[1] if teams[0].upper() == player_team.upper() else teams[0]
    except Exception as e:
        print(f"Error parsing matchup from '{game_info}': {e}")
        return None

def get_detailed_opponent_defense(opponent_tricode, season=None):
    """
    Retrieve the opponent's defensive rating from DefenseHub.
    """
    if season is None:
        season = get_current_season()
    try:
        dh = defensehub.DefenseHub(season=season)
        df = dh.defense_hub_stat4.get_data_frame()
        row = df[df['TEAM_ABBREVIATION'] == opponent_tricode.upper()]
        return row['TM_DEF_RATING'].iloc[0] if not row.empty else None
    except Exception as e:
        print(f"Error fetching opponent defense for {opponent_tricode}: {e}")
        return None

def get_opponent_defensive_boxscore(game_id, player_id):
    """
    Retrieve opponent defensive boxscore info (e.g., allowed points) using BoxScoreDefensiveV2.
    """
    try:
        bsd = boxscoredefensivev2.BoxScoreDefensiveV2(game_id=game_id)
        df = bsd.player_stats.get_data_frame()
        data = df[df['personId'] == int(player_id)]
        return data['playerPoints'].iloc[0] if not data.empty else None
    except Exception as e:
        print(f"Error fetching defensive boxscore for player {player_id}: {e}")
        return None

def get_fantasy_points_per_minute(player_id, season=None):
    """
    Compute fantasy points per minute (FPPM) using PlayerFantasyProfile.
    """
    if season is None:
        season = get_current_season()
    try:
        profile = playerfantasyprofile.PlayerFantasyProfile(player_id=player_id, season=season)
        df = profile.overall.get_data_frame()
        df['NBA_FANTASY_PTS'] = pd.to_numeric(df['NBA_FANTASY_PTS'], errors='coerce')
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        avg_pts = df['NBA_FANTASY_PTS'].mean()
        avg_min = df['MIN'].mean()
        return avg_pts / avg_min if avg_min > 0 else 0.0
    except Exception as e:
        print(f"Error computing FPPM for player {player_id}: {e}")
        return 0.0

def get_rolling_fantasy_avg(player_id, season=None, n=5):
    """
    Compute the rolling average of fantasy points over the last n games using PlayerGameLogs.
    """
    if season is None:
        season = get_current_season()
    try:
        logs = playergamelogs.PlayerGameLogs(player_id_nullable=str(player_id), season_nullable=season)
        df = logs.player_game_logs
        col = "NBA_FANTASY_PTS" if "NBA_FANTASY_PTS" in df.columns else "PTS"
        return df.tail(n)[col].mean() if not df.tail(n).empty else 0.0
    except Exception as e:
        print(f"Error computing rolling fantasy avg for player {player_id}: {e}")
        return 0.0

def compute_efg(stats):
    """
    Compute effective field goal percentage (eFG%):
      eFG% = (FGM + 0.5 * FG3M) / FGA.
    """
    fgm = stats.get('FGM', 0.0)
    fg3m = stats.get('FG3M', 0.0)
    fga = stats.get('FGA', 0.0)
    return (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0

def compute_true_shooting_attempts(stats):
    """
    Compute true shooting attempts (TSA):
      TSA = FGA + 0.44 * FTA.
    """
    fga = stats.get('FGA', 0.0)
    fta = stats.get('FTA', 0.0)
    return fga + 0.44 * fta

# Additional endpoints for enhanced features

def get_home_away_fantasy_ratio(player_id, season=None):
    """
    Compute the ratio of fantasy points at home versus away using PlayerDashboardByGeneralSplits.
    """
    if season is None:
        season = get_current_season()
    try:
        dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(player_id=player_id, season=season)
        df = dashboard.location_player_dashboard.get_data_frame()
        home = df[df["GROUP_VALUE"]=="Home"]["NBA_FANTASY_PTS"].mean()
        away = df[df["GROUP_VALUE"]=="Away"]["NBA_FANTASY_PTS"].mean()
        return home / away if away and away > 0 else 1.0
    except Exception as e:
        print(f"Error computing home/away fantasy ratio: {e}")
        return 1.0

def get_team_lineup_efficiency(team_id, game_id):
    """
    Compute team lineup efficiency (average plus/minus) using TeamDashLineups.
    """
    try:
        lineups = teamdashlineups.TeamDashLineups(team_id=team_id, game_id_nullable=game_id)
        df = lineups.overall.get_data_frame()
        return df["PLUS_MINUS"].mean() if "PLUS_MINUS" in df.columns else 0.0
    except Exception as e:
        print(f"Error computing team lineup efficiency: {e}")
        return 0.0

def get_player_estimated_metrics(player_id, season=None):
    """
    Retrieve estimated metrics (e.g., net rating, usage percentage) from PlayerEstimatedMetrics.
    """
    if season is None:
        season = get_current_season()
    try:
        est = playerestimatedmetrics.PlayerEstimatedMetrics(player_id=player_id, season=season)
        df = est.player_estimated_metrics.get_data_frame()
        net = df["E_NET_RATING"].iloc[0] if "E_NET_RATING" in df.columns and not df.empty else 0.0
        usg = df["E_USG_PCT"].iloc[0] if "E_USG_PCT" in df.columns and not df.empty else 0.0
        return {"E_NET_RATING": net, "E_USG_PCT": usg}
    except Exception as e:
        print(f"Error retrieving estimated metrics: {e}")
        return {"E_NET_RATING": 0.0, "E_USG_PCT": 0.0}

def get_player_performance_std(player_id, season=None):
    """
    Compute the standard deviation of a player's fantasy points over recent games using PlayerGameLogs.
    """
    if season is None:
        season = get_current_season()
    try:
        logs = playergamelogs.PlayerGameLogs(player_id_nullable=str(player_id), season_nullable=season)
        df = logs.player_game_logs
        col = "NBA_FANTASY_PTS" if "NBA_FANTASY_PTS" in df.columns else "PTS"
        return df[col].std() if not pd.isna(df[col].std()) else 0.0
    except Exception as e:
        print(f"Error computing performance std: {e}")
        return 0.0

# ------------------ Contextual Factors ------------------

def get_home_away_factor_from_gamelog(gamelog_df, game_info, player_team):
    """
    Compute a multiplier based on historical efficiency differences between home and away games.
    """
    if 'MATCHUP' not in gamelog_df.columns:
        return 1.0
    def efficiency(row):
        return (row['PTS'] + row['REB'] + row['AST'] + row['STL'] + row['BLK'] - row['TOV']) / row['MIN'] if row['MIN'] > 0 else 0.0
    gamelog_df = gamelog_df.copy()
    gamelog_df['efficiency'] = gamelog_df.apply(efficiency, axis=1)
    overall = gamelog_df['efficiency'].mean() if len(gamelog_df) > 0 else 1.0
    home = gamelog_df[gamelog_df['MATCHUP'].str.contains("vs")]['efficiency'].mean()
    away = gamelog_df[gamelog_df['MATCHUP'].str.contains("@")]['efficiency'].mean()
    if "vs" in game_info:
        return home / overall if overall > 0 else 1.0
    elif "@" in game_info:
        return away / overall if overall > 0 else 1.0
    else:
        return 1.0

def get_rest_factor(gamelog_df):
    """
    Compute the rest factor as the ratio of the most recent rest days to the average rest days.
    """
    try:
        gamelog_df = gamelog_df.copy()
        gamelog_df['GAME_DATE'] = pd.to_datetime(gamelog_df['GAME_DATE'])
        df_sorted = gamelog_df.sort_values(by='GAME_DATE')
        df_sorted['RestDays'] = df_sorted['GAME_DATE'].diff().dt.days
        avg = df_sorted['RestDays'].mean()
        return df_sorted['RestDays'].iloc[-1] / avg if len(df_sorted) >= 2 and avg and avg > 0 else 1.0
    except Exception as e:
        print(f"Error computing rest factor: {e}")
        return 1.0

def get_lineup_factor(game_id, player_id):
    """
    Compute the lineup factor using BoxScoreAdvancedV3.
    """
    try:
        adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        df = adv.player_stats.get_data_frame()
        team_avg = df['PLUS_MINUS'].mean()
        player_pm = df[df['PERSON_ID'] == int(player_id)]['PLUS_MINUS']
        if player_pm.empty or team_avg == 0:
            return 1.0
        return 1 + ((player_pm.iloc[0] - team_avg) / abs(team_avg))
    except Exception as e:
        print(f"Error computing lineup factor: {e}")
        return 1.0

def get_opponent_ml_adjustment(opponent_tricode, season=None):
    """
    Compute opponent adjustment factor using league and opponent defensive ratings.
    """
    if season is None:
        season = get_current_season()
    league_avg = get_league_average_def_rating(season)
    opp_def = get_detailed_opponent_defense(opponent_tricode, season)
    if league_avg and opp_def and opp_def != 0:
        return league_avg / opp_def
    return 1.0

def calculate_advanced_metrics(stats, gamelog_df, team_pace, season):
    """
    Compute advanced metrics:
    - PER (simplified): (PTS+REB+AST+STL+BLK-TOV) / MIN
    - True Shooting Percentage (TS%): PTS / (2*(FGA+0.44*FTA))
    - Assist-to-Turnover Ratio: AST / TOV (or AST if TOV is zero)
    - Simplified DWS: ((STL+BLK)/MIN) adjusted by team pace.
    """
    minutes = stats.get('MIN', 0.0)
    per = (stats['PTS'] + stats['REB'] + stats['AST'] + stats['STL'] + stats['BLK'] - stats['TOV']) / minutes if minutes > 0 else 0.0
    fga = stats.get('FGA', 0.0)
    fta = stats.get('FTA', 0.0)
    denom = 2 * (fga + 0.44 * fta)
    ts = stats['PTS'] / denom if denom > 0 else 0.0
    ast_tov = stats['AST'] / stats['TOV'] if stats['TOV'] > 0 else stats['AST']
    league_avg = get_league_average_pace(season)
    dws = ((stats['STL'] + stats['BLK']) / minutes) * (team_pace / league_avg) if minutes > 0 and league_avg > 0 else 0.0
    return per, ts, ast_tov, dws

# ------------------ FEATURE EXTRACTION ------------------
def calculate_features(player_name, player_team, game_info):
    """
    Compute a complete feature vector for a player using data from various nba_api endpoints.
    """
    season = get_current_season()
    nba_player_id = get_nba_player_id(player_name)
    if not nba_player_id:
        return None
    stats, gamelog_df = get_player_season_stats(nba_player_id, season)
    if gamelog_df is None or gamelog_df.empty or stats is None:
        return None
    features = stats.copy()  # Start with base stats

    team_pace = get_team_pace(player_team, season)
    if not team_pace:
        team_pace = 100.0

    # Advanced metrics
    per, ts, ast_tov, dws = calculate_advanced_metrics(stats, gamelog_df, team_pace, season)
    features['PER'] = per
    features['TS'] = ts
    features['AST_TOV'] = ast_tov
    features['DWS'] = dws

    # Additional shooting metrics
    features['eFG'] = compute_efg(stats)
    features['TSA'] = compute_true_shooting_attempts(stats)

    recent_game_id = gamelog_df.iloc[0]['GAME_ID']
    usage, _ = get_player_usage_boxscore(nba_player_id, recent_game_id)
    features['Usage'] = usage if usage is not None else 0.0

    opponent_tricode = parse_opponent_from_game_info(game_info, player_team)
    detailed_def = get_detailed_opponent_defense(opponent_tricode, season) if opponent_tricode else None
    features['OpponentDefense'] = detailed_def if detailed_def is not None else 1.0
    allowed_pts = get_opponent_defensive_boxscore(recent_game_id, nba_player_id)
    features['AllowedPoints'] = allowed_pts if allowed_pts is not None else 0.0

    features['TeamPace'] = team_pace
    features['InactiveAdjustment'] = get_team_inactive_adjustment(player_team, nba_player_id, recent_game_id)
    features['HomeAwayFactor'] = get_home_away_factor_from_gamelog(gamelog_df, game_info, player_team)
    features['RestFactor'] = get_rest_factor(gamelog_df)
    features['LineupFactor'] = get_lineup_factor(recent_game_id, nba_player_id)
    features['OpponentMLAdj'] = get_opponent_ml_adjustment(opponent_tricode, season)

    features['FPPM'] = get_fantasy_points_per_minute(nba_player_id, season)
    features['HomeAwayFantasyRatio'] = get_home_away_fantasy_ratio(nba_player_id, season)

    team_id = gamelog_df.iloc[0]['TEAM_ID'] if 'TEAM_ID' in gamelog_df.columns else None
    features['TeamLineupEfficiency'] = get_team_lineup_efficiency(team_id, recent_game_id) if team_id else 0.0

    est = get_player_estimated_metrics(nba_player_id, season)
    features['E_NET_RATING'] = est.get("E_NET_RATING", 0.0)
    features['E_USG_PCT'] = est.get("E_USG_PCT", 0.0)

    features['PtsStd'] = get_player_performance_std(nba_player_id, season)
    features['RollingFantasyAvg'] = get_rolling_fantasy_avg(nba_player_id, season, n=5)

    return features

def create_feature_dataframe_from_csv(df):
    """
    Loop through each row of the input CSV and compute the feature vector for each player.
    Returns a DataFrame of feature vectors.
    """
    feature_rows = []
    for _, row in df.iterrows():
        player_name = row['Name']
        player_team = row['TeamAbbrev']
        game_info = row['Game Info']
        feat = calculate_features(player_name, player_team, game_info)
        if feat is not None:
            feature_rows.append(feat)
    return pd.DataFrame(feature_rows) if feature_rows else None
