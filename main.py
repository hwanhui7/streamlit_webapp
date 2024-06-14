import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)


def view_team_stat(team_info):
    if team_info != 'Team':
        col1, col2 = st.columns([2, 1])

        with col1:
            image_path = f'{team_info}.png'
            st.image(image_path, width=400)

        with col2:
            team_stats = ranked_teams[ranked_teams['common_name'] == team_info].iloc[0]
            st.subheader('Team Stats')
            st.markdown(f"""
                            <p style="font-size: 20px; line-height: 1.5;">Position: <strong>{team_stats['rank']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Wins: <strong>{team_stats['wins']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Draws: <strong>{team_stats['draws']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Losses: <strong>{team_stats['losses']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Points: <strong>{team_stats['points']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Goals Scored: <strong>{team_stats['goals_scored']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Goals Conceded: <strong>{team_stats['goals_conceded']}</p>
                            <p style="font-size: 20px; line-height: 1.5;">Goal Difference: <strong>{team_stats['goal_difference']}</p>
                        """, unsafe_allow_html=True)


def get_match_info(game_week, team):
    match = matches_data[(matches_data['Game Week'] == game_week) &
                         ((matches_data['home_team_name'] == team) |
                          (matches_data['away_team_name'] == team))]
    if not match.empty:
        return match.iloc[0]
    else:
        return None


# Load data
matches_data = load_data('matches.csv')
teams_data = load_data('teams.csv')

teams_data['points'] = teams_data['wins'] * 3 + teams_data['draws']
teams_data['goal_difference'] = teams_data['goals_scored'] - teams_data['goals_conceded']
ranked_teams = teams_data.sort_values(by=['points', 'goal_difference', 'goals_scored'], ascending=[False, False, False])
ranked_teams['rank'] = range(1, len(ranked_teams) + 1)

st.sidebar.header('Select Info Type')

info_type = st.sidebar.radio('Choose Information to Display:', ['Team Info', 'Match Info', 'Predict Match'])

if info_type == 'Team Info':
    st.title('Team Info')
    st.sidebar.header('Team Info Filters')
    team_info = st.sidebar.selectbox('Team Name', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

    if team_info != 'Team':
        st.header(team_info)
        st.write(' ')
        st.write(' ')
        view_team_stat(team_info)

elif info_type == 'Match Info':
    st.title('Match Info')
    st.write(' ')
    st.sidebar.header('Match Info Filters')
    game_week = st.sidebar.selectbox('Game Week', ['Week'] + sorted(matches_data['Game Week'].unique().tolist()))
    team = st.sidebar.selectbox('Match Team', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

    match_info = get_match_info(game_week, team)

    if match_info is not None:
        col1, col2, col3, col4, col5 = st.columns([1, 5, 2, 5, 1])

        with col1:
            home_image_path = f'{match_info["home_team_name"]}.png'
            st.image(home_image_path, width=50)

        with col2:
            st.markdown(f"<h3 style='text-align: center;'>{match_info['home_team_name']}</h3>", unsafe_allow_html=True)

        with col3:
            st.markdown(
                f"<h3 style='text-align: center;'>{match_info['home_team_goal_count']} - {match_info['away_team_goal_count']}</h3>",
                unsafe_allow_html=True)

        with col4:
            st.markdown(f"<h3 style='text-align: center;'>{match_info['away_team_name']}</h3>", unsafe_allow_html=True)

        with col5:
            away_image_path = f'{match_info["away_team_name"]}.png'
            st.image(away_image_path, width=50)


        stadium_name = match_info['stadium_name']
        cleaned_stadium_name = re.sub(r'\s*\([^)]*\)', '', stadium_name)

        tab1, tab2 = st.tabs(["Match Info", "Stats"])

        with tab1:
            st.subheader("Match Info")
            st.write(f"Match Date & Time: {match_info['date_GMT']}")
            st.write(f"Home Team: {match_info['home_team_name']}")
            st.write(f"Away Team: {match_info['away_team_name']}")
            st.write(f"Stadium: {cleaned_stadium_name}")
            st.write(f"Referee: {match_info['referee']}")

        with tab2:
            st.subheader("Stats")
            stats_data = {
                "Possession": [match_info['home_team_possession'], match_info['away_team_possession']],
                "Shot": [match_info['home_team_shots'], match_info['away_team_shots']],
                "Shot on Target": [match_info['home_team_shots_on_target'], match_info['away_team_shots_on_target']],
                "Corner": [match_info['home_team_corner_count'], match_info['away_team_corner_count']],
                "Foul": [match_info['home_team_fouls'], match_info['away_team_fouls']],
                "Yellow Card": [match_info['home_team_yellow_cards'], match_info['away_team_yellow_cards']],
                "Red Card": [match_info['home_team_red_cards'], match_info['away_team_red_cards']]
            }

            def plot_stats(data, categories):
                fig, axs = plt.subplots(7, 1, figsize=(7, 4.5), tight_layout=True)  # Adjusted figsize for flatter graphs

                for i, category in enumerate(categories):
                    values = data[category]
                    total = sum(values)

                    if total == 0:
                        normalized_team1_percentage = 50
                        normalized_team2_percentage = 50
                        label1 = label2 = '0'
                    else:
                        team1_percentage = values[0] / total * 100
                        team2_percentage = values[1] / total * 100
                        scale = 100 / total
                        normalized_team1_percentage = values[0] * scale
                        normalized_team2_percentage = values[1] * scale
                        label1 = f'{team1_percentage:.1f}%' if category == "Possession" else f'{values[0]}'
                        label2 = f'{team2_percentage:.1f}%' if category == "Possession" else f'{values[1]}'

                    axs[i].barh(0, normalized_team1_percentage, color='pink', label=match_info['home_team_name'] if i == 0 else "")
                    axs[i].barh(0, normalized_team2_percentage, left=normalized_team1_percentage, color='skyblue', label=match_info['away_team_name'] if i == 0 else "")
                    axs[i].set_xlim(0, 100)

                    axs[i].text(normalized_team1_percentage / 2, 0, label1, va='center', ha='center', color='black', fontsize=10)
                    axs[i].text(normalized_team1_percentage + normalized_team2_percentage / 2, 0, label2, va='center', ha='center', color='black', fontsize=10)

                    axs[i].axis('off')
                    axs[i].set_title(category, loc='left')

                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

                st.pyplot(fig)

            plot_stats(stats_data, list(stats_data.keys()))


elif info_type == 'Predict Match':
    st.title('Predict Match')
    st.sidebar.header('Match Prediction Filters')

    home_team = st.sidebar.selectbox('Home Team', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))
    away_team = st.sidebar.selectbox('Away Team', ['Team'] + sorted(matches_data['away_team_name'].unique().tolist()))

    if home_team != 'Team' and away_team != 'Team':
        # 데이터 로드
        data = matches_data


        def predict_match(home_team, away_team, specific_weight=5, general_weight=1):
            specific_matches = data[(data['home_team_name'] == home_team) & (data['away_team_name'] == away_team)]
            general_home_matches = data[(data['home_team_name'] == home_team) & (data['away_team_name'] != away_team)]
            general_away_matches = data[(data['away_team_name'] == away_team) & (data['home_team_name'] != home_team)]

            weighted_specific_matches = pd.concat([specific_matches] * specific_weight, ignore_index=True)

            weighted_home_matches = pd.concat([weighted_specific_matches, general_home_matches], ignore_index=True)
            weighted_away_matches = pd.concat([weighted_specific_matches, general_away_matches], ignore_index=True)

            home_team_stats = weighted_home_matches.groupby('home_team_name').agg({
                'home_team_possession': 'mean',
                'home_team_shots': 'mean',
                'home_team_goal_count': 'mean'
            }).reset_index()

            away_team_stats = weighted_away_matches.groupby('away_team_name').agg({
                'away_team_possession': 'mean',
                'away_team_shots': 'mean',
                'away_team_goal_count': 'mean'
            }).reset_index()

            if home_team_stats.empty or away_team_stats.empty:
                return "Team data not available."

            home_X = weighted_home_matches[['home_team_possession', 'home_team_shots']]
            home_y = weighted_home_matches['home_team_goal_count']
            home_model = RandomForestRegressor(n_estimators=100)
            home_model.fit(home_X, home_y)

            away_X = weighted_away_matches[['away_team_possession', 'away_team_shots']]
            away_y = weighted_away_matches['away_team_goal_count']
            away_model = RandomForestRegressor(n_estimators=100)
            away_model.fit(away_X, away_y)

            home_possession = home_team_stats['home_team_possession'].values[0]
            away_possession = away_team_stats['away_team_possession'].values[0]
            home_shots = home_team_stats['home_team_shots'].values[0]
            away_shots = away_team_stats['away_team_shots'].values[0]

            home_input = pd.DataFrame([[home_possession, home_shots]],
                                      columns=['home_team_possession', 'home_team_shots'])
            away_input = pd.DataFrame([[away_possession, away_shots]],
                                      columns=['away_team_possession', 'away_team_shots'])

            home_goals = home_model.predict(home_input)[0]
            away_goals = away_model.predict(away_input)[0]

            return {
                'home_possession': 100 * home_possession / (home_possession + away_possession),
                'away_possession': 100 * away_possession / (home_possession + away_possession),
                'home_shots': home_shots,
                'away_shots': away_shots,
                'home_goals': home_goals,
                'away_goals': away_goals
            }


        prediction = predict_match(home_team, away_team)

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            st.markdown(f"<h3 style='text-align: center;'>{home_team}</h3>", unsafe_allow_html=True)
            home_image_path = f'{home_team}.png'
            st.image(home_image_path, width=270)


        with col2:
            st.markdown(
                f"<h1 style='text-align: center;'>{round(prediction['home_goals'])} - {round(prediction['away_goals'])}</h1>",
                unsafe_allow_html=True)

        with col3:
            st.markdown(f"<h3 style='text-align: center;'>{away_team}</h3>", unsafe_allow_html=True)
            away_image_path = f'{away_team}.png'
            st.image(away_image_path, width=270)


        stats_data = {
            "Goals": [prediction['home_goals'], prediction['away_goals']],
            "Possession": [prediction['home_possession'], prediction['away_possession']],
            "Shots": [prediction['home_shots'], prediction['away_shots']]
        }


        def plot_stats(data, categories):
            fig, axs = plt.subplots(len(categories), 1, figsize=(7, 3), tight_layout=True)

            for i, category in enumerate(categories):
                values = data[category]
                total = sum(values)

                if total == 0:
                    normalized_team1_percentage = 50
                    normalized_team2_percentage = 50
                    label1 = label2 = '0'
                else:
                    team1_percentage = values[0] / total * 100
                    team2_percentage = values[1] / total * 100
                    scale = 100 / total
                    normalized_team1_percentage = values[0] * scale
                    normalized_team2_percentage = values[1] * scale
                    label1 = f'{team1_percentage:.1f}%' if category == "Possession" else f'{values[0]:.2f}'
                    label2 = f'{team2_percentage:.1f}%' if category == "Possession" else f'{values[1]:.2f}'

                axs[i].barh(0, normalized_team1_percentage, color='orange', label=home_team if i == 0 else "")
                axs[i].barh(0, normalized_team2_percentage, left=normalized_team1_percentage, color='yellowgreen',
                            label=away_team if i == 0 else "")
                axs[i].set_xlim(0, 100)

                axs[i].text(normalized_team1_percentage / 2, 0, label1, va='center', ha='center', color='black',
                            fontsize=10)
                axs[i].text(normalized_team1_percentage + normalized_team2_percentage / 2, 0, label2, va='center',
                            ha='center', color='black', fontsize=10)

                axs[i].axis('off')
                axs[i].set_title(category, loc='left')

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

            st.pyplot(fig)

        plot_stats(stats_data, list(stats_data.keys()))

