import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
@st.cache_data  # 데이터 캐싱
def load_data(filepath):
    return pd.read_csv(filepath)

def view_team_stat(team_info):
    if team_info != 'Team':
        # 팀 로고 및 스탯 표시
        col1, col2 = st.columns([2, 1])

        # 왼쪽 컬럼에 팀 로고 표시
        with col1:
            image_path = f'{team_info}.png'
            st.image(image_path, width=400)

        # 오른쪽 컬럼에 팀 통계 표시
        with col2:
            team_stats = ranked_teams[ranked_teams['common_name'] == team_info].iloc[0]
            st.subheader('Team Stats')
            st.write(f'Position: {team_stats["rank"]}')
            st.write(f'Wins: {team_stats["wins"]}')
            st.write(f'Draws: {team_stats["draws"]}')
            st.write(f'Losses: {team_stats["losses"]}')
            st.write(f'Points: {team_stats["points"]}')
            st.write(f'Goals For: {team_stats["goals_scored"]}')
            st.write(f'Goals Against: {team_stats["goals_conceded"]}')
            st.write(f'Goals Difference: {team_stats["goals_difference"]}')

# ----------------------------------------------------------------------------------------------------------------------

matches_data = load_data('matches.csv')
teams_data = load_data('teams.csv')

st.title('Premier League Data Analysis')
st.sidebar.header('Filters')

team_info = st.sidebar.selectbox('Team Info', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

teams_data['points'] = teams_data['wins'] * 3 + teams_data['draws']
teams_data['goals_difference'] = teams_data['goals_scored'] - teams_data['goals_conceded']


# 순위를 결정하기 위한 정렬
ranked_teams = teams_data.sort_values(by=['points', 'goals_difference', 'goals_scored'], ascending=[False, False, False])
ranked_teams['rank'] = range(1, len(ranked_teams) + 1)

if team_info == 'Team':
    pass
else:
    st.header(team_info)
    view_team_stat(team_info)