import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

matches_data = load_data('matches.csv')
teams_data = load_data('teams.csv')

teams_data['points'] = teams_data['wins'] * 3 + teams_data['draws']
teams_data['goal_difference'] = teams_data['goals_scored'] - teams_data['goals_conceded']

ranked_teams = teams_data.sort_values(by=['points', 'goal_difference', 'goals_scored'], ascending=[False, False, False])
ranked_teams['rank'] = range(1, len(ranked_teams) + 1)

st.sidebar.markdown("""
    <div style="font-size: 24px; line-height: 1.5;">
        <strong>20 Premier League clubs<br>Match Record Analysis<br>Expected Match Results
    </div>
""", unsafe_allow_html=True)

st.sidebar.write('')

info_type = st.sidebar.radio('Choose Information to Display:', ['Team Info', 'Match Info', 'Predict Match', 'All Stat Analysis', 'Team Stat Analysis'])


if info_type == 'Team Info':
    st.title('Team Info')
    st.sidebar.header('Team Info Filters')

    team_info = st.sidebar.selectbox('Team Name', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

    if team_info != 'Team':
        st.header(team_info)

        st.write(' ')
        st.write(' ')

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


elif info_type == 'Match Info':
    st.title('Match Info')

    st.write(' ')

    st.sidebar.header('Match Info Filters')

    game_week = st.sidebar.selectbox('Game Week', ['Week'] + sorted(matches_data['Game Week'].unique().tolist()))
    team = st.sidebar.selectbox('Match Team', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

    def get_match_info(game_week, team):
        match = matches_data[(matches_data['Game Week'] == game_week) &
                             ((matches_data['home_team_name'] == team) |
                              (matches_data['away_team_name'] == team))]

        if not match.empty:
            return match.iloc[0]

        else:
            return None

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
    st.sidebar.header('Predict Match Filters')

    home_team = st.sidebar.selectbox('Home Team', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))
    away_team = st.sidebar.selectbox('Away Team', ['Team'] + sorted(matches_data['away_team_name'].unique().tolist()))

    if home_team != 'Team' and away_team != 'Team':
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


elif info_type == 'All Stat Analysis':
    st.title('All Stat Analysis')
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["점유율-골", "슈팅-골", "유효슈팅-골", "슈팅-유효슈팅", "점유율-슈팅", "점유율-파울", "파울-카드"])
    data = matches_data

    with tab1:
        st.subheader('홈 팀과 원정 팀의 점유율에 따른 득점 수')

        home_correlation = data['home_team_goal_count'].corr(data['home_team_possession'])
        away_correlation = data['away_team_goal_count'].corr(data['away_team_possession'])

        X_home = sm.add_constant(data['home_team_possession'])
        model_home = sm.OLS(data['home_team_goal_count'], X_home).fit()
        home_slope = model_home.params['home_team_possession']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_possession'])
        model_away = sm.OLS(data['away_team_goal_count'], X_away).fit()
        away_slope = model_away.params['away_team_possession']
        away_intercept = model_away.params['const']

        max_possession = max(data['home_team_possession'].max(), data['away_team_possession'].max())
        max_goals = max(data['home_team_goal_count'].max(), data['away_team_goal_count'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_possession', y='home_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Possession vs Goals\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Possession')
        plt.ylabel('Home Team Goals')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_possession', y='away_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Possession vs Goals\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Possession')
        plt.ylabel('Away Team Goals')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.tight_layout()

        st.pyplot(plt)

        st.markdown("""
            <div style="font-size: 13px; line-height: 1.8;">
                홈 팀의 그래프에서 볼 수 있듯이, 점유율이 높아질수록 득점이 증가하는 경향이 있다. 이는 홈 팀이 경기에서 더 많은 점유율을 
                가지는 경우 더 공격적인 플레이를 하고, 더 많은 득점을 올릴 수 있는 기회를 가질 수 있음을 보여준다. 홈 경기의 이점(팬의 응원, 
                익숙한 환경 등)이 점유율과 득점에 긍정적인 영향을 미칠 수 있다.<br>
                원정 팀의 그래프에서는 점유율이 높을수록 득점이 증가하는 경향이 있지만, 홈 팀에 비해 그 기울기가 약간 낮다. 이는 원정 팀이 
                높은 점유율을 유지하기 어렵고, 득점을 올리는 데 더 많은 도전이 필요함을 의미한다. 원정 경기에서는 낯선 환경, 상대 팀 팬의 
                압박, 긴 이동 거리 등 여러 가지 불리한 요인이 있을 수 있다.<br>
                결론적으로 홈 팀은 점유율이 높을수록 득점이 증가하는 경향이 더 뚜렷하며 홈 어드밴티지를 활용하여 더 공격적인 플레이를 펼칠 
                가능성이 크고, 원정 팀은 점유율이 높을수록 득점이 증가하지만, 홈 팀에 비해 그 영향이 약간 적으므로 원정 경기의 불리한 요인을 
                극복하고 효율적인 공격 전술을 사용해야 한다.
            </div>
        """, unsafe_allow_html=True)

        st.sidebar.write('')

    with tab2:
        st.subheader("홈 팀과 원정 팀의 슈팅 수에 따른 득점 수")

        home_correlation = data['home_team_goal_count'].corr(data['home_team_shots'])
        away_correlation = data['away_team_goal_count'].corr(data['away_team_shots'])

        X_home = sm.add_constant(data['home_team_shots'])
        model_home = sm.OLS(data['home_team_goal_count'], X_home).fit()
        home_slope = model_home.params['home_team_shots']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_shots'])
        model_away = sm.OLS(data['away_team_goal_count'], X_away).fit()
        away_slope = model_away.params['away_team_shots']
        away_intercept = model_away.params['const']

        max_shots = max(data['home_team_shots'].max(), data['away_team_shots'].max())
        max_goals = max(data['home_team_goal_count'].max(), data['away_team_goal_count'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_shots', y='home_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Shots vs Goals\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Shots')
        plt.ylabel('Home Team Goals')

        plt.xlim(-0.02 * max_shots, 1.02 * max_shots)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_shots', y='away_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Shots vs Goals\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Shots')
        plt.ylabel('Away Team Goals')

        plt.xlim(-0.02 * max_shots, 1.02 * max_shots)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 슈팅 수가 증가할수록 득점이 더 많이 증가하는 경향이 있다. 홈 팀이 경기에서 공격적으로 플레이할 
                        가능성이 크며, 이는 팬들의 응원, 익숙한 환경 등으로 인해 슈팅의 성공률이 높아질 수 있음을 나타낸다. Y절편이 0.186인 
                        것은 홈 팀이 슈팅을 거의 하지 않아도 일정 득점을 기록할 가능성이 있음을 보여준다.<br>                        
                        원정 팀의 경우 슈팅 수가 증가할수록 득점이 증가하는 경향이 있지만, 홈 팀에 비해 Y절편이 0.072로 낮아 득점을 올리는 
                        것이 더 어려울 수 있음을 의미한다. 원정 팀의 상관 계수는 0.363으로, 홈 팀의 상관 계수인 0.398보다 조금 낮다. 이는 
                        원정 팀이 슈팅 시도에 더 많은 노력을 기울여야 득점으로 연결될 가능성이 높다는 것을 나타낸다.<br>                        
                        결론적으로 홈 팀은 슈팅 수가 많아질수록 득점이 증가하는 경향이 더 뚜렷하고, 홈 어드밴티지를 통해 더 공격적인 플레이를 
                        펼치며 슈팅이 득점으로 연결될 가능성이 크고, 원정 팀은 슈팅 수가 많아질수록 득점이 증가하지만 홈 팀에 비해 그 영향이 
                        약간 적기 때문에 원정 경기의 불리한 요인을 극복하고 효율적인 슈팅 전술을 사용해야 한다.
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')

    with tab3:
        st.subheader('홈 팀과 원정 팀의 유효슈팅 수에 따른 득점 수')

        home_correlation = data['home_team_goal_count'].corr(data['home_team_shots_on_target'])
        away_correlation = data['away_team_goal_count'].corr(data['away_team_shots_on_target'])

        X_home = sm.add_constant(data['home_team_shots_on_target'])
        model_home = sm.OLS(data['home_team_goal_count'], X_home).fit()
        home_slope = model_home.params['home_team_shots_on_target']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_shots_on_target'])
        model_away = sm.OLS(data['away_team_goal_count'], X_away).fit()
        away_slope = model_away.params['away_team_shots_on_target']
        away_intercept = model_away.params['const']

        max_shots_on_target = max(data['home_team_shots_on_target'].max(), data['away_team_shots_on_target'].max())
        max_goals = max(data['home_team_goal_count'].max(), data['away_team_goal_count'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_shots_on_target', y='home_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Shots on Target vs Goals\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Shots on Target')
        plt.ylabel('Home Team Goals')

        plt.xlim(-0.02 * max_shots_on_target, 1.02 * max_shots_on_target)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_shots_on_target', y='away_team_goal_count', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Shots on Target vs Goals\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f} Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Shots on Target')
        plt.ylabel('Away Team Goals')

        plt.xlim(-0.02 * max_shots_on_target, 1.02 * max_shots_on_target)
        plt.ylim(-0.02 * max_goals, 1.02 * max_goals)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 유효 슈팅 수가 증가할수록 득점이 더 많이 증가하는 경향이 있다. 홈 팀이 경기에서 공격적으로 
                        플레이할 가능성이 크며, 이는 팬들의 응원, 익숙한 환경 등으로 인해 유효 슈팅의 성공률이 높아질 수 있음을 나타낸다. 
                        상관 계수가 0.593으로 비교적 높아, 유효 슈팅 수와 득점 수 사이에 강한 양의 상관 관계가 있음을 나타낸다.<br>                        
                        원정 팀의 경우 유효 슈팅 수가 증가할수록 득점이 증가하는 경향이 있지만, 원정 팀의 상관 계수는 0.545로, 홈 팀의 상관 
                        계수인 0.593보다 조금 낮다. 이는 원정 팀이 유효 슈팅 시도에 더 많은 노력을 기울여야 득점으로 연결될 가능성이 높다는 
                        것을 나타낸다.<br>                        
                        결론적으로 홈 팀은 유효 슈팅 수가 많아질수록 득점이 증가하는 경향이 더 뚜렷하고, 홈 어드밴티지를 통해 더 공격적인 
                        플레이를 펼치며 유효 슈팅이 득점으로 연결될 가능성이 크다. 원정 팀은 유효 슈팅 수가 많아질수록 득점이 증가하지만 홈 
                        팀에 비해 그 영향이 약간 적기 때문에 원정 경기의 불리한 요인을 극복하고 효율적인 슈팅 전술을 사용해야 한다.<br>
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')


    with tab4:
        st.subheader("홈 팀과 원정 팀의 슈팅 수에 따른 유효슈팅 수")

        home_correlation = data['home_team_shots'].corr(data['home_team_shots_on_target'])
        away_correlation = data['away_team_shots'].corr(data['away_team_shots_on_target'])

        X_home = sm.add_constant(data['home_team_shots'])
        model_home = sm.OLS(data['home_team_shots_on_target'], X_home).fit()
        home_slope = model_home.params['home_team_shots']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_shots'])
        model_away = sm.OLS(data['away_team_shots_on_target'], X_away).fit()
        away_slope = model_away.params['away_team_shots']
        away_intercept = model_away.params['const']

        max_shots = max(data['home_team_shots'].max(), data['away_team_shots'].max())
        max_shots_on_target = max(data['home_team_shots_on_target'].max(), data['away_team_shots_on_target'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_shots', y='home_team_shots_on_target', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Shots vs Shots on Target\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Shots')
        plt.ylabel('Home Team Shots on Target')

        plt.xlim(-0.02 * max_shots, 1.02 * max_shots)
        plt.ylim(-0.02 * max_shots_on_target, 1.02 * max_shots_on_target)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_shots', y='away_team_shots_on_target', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Shots vs Shots on Target\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Shots')
        plt.ylabel('Away Team Shots on Target')

        plt.xlim(-0.02 * max_shots, 1.02 * max_shots)
        plt.ylim(-0.02 * max_shots_on_target, 1.02 * max_shots_on_target)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 슈팅 수가 증가할수록 유효 슈팅 수가 더 많이 증가하는 경향이 있다. 이는 홈 팀이 경기에서 
                        공격적으로 플레이할 가능성이 크다는 것을 나타낸다. 상관 계수가 0.761로 매우 높아, 슈팅 수와 유효 슈팅 수 사이에 
                        강한 양의 상관 관계가 있음을 나타낸다.<br>                        
                        원정 팀의 경우 슈팅 수가 증가할수록 유효 슈팅 수가 증가하는 경향이 있지만, 홈 팀에 비해 Y절편이 낮아 유효 슈팅을 
                        만드는 것이 더 어려울 수 있음을 의미한다. 원정 팀의 상관 계수는 0.744로, 홈 팀의 상관 계수보다 조금 낮은데 이는 
                        원정 팀이 유효 슈팅 시도에 더 많은 노력을 기울여야 득점으로 연결될 가능성이 높음을 나타낸다.<br>                        
                        결론적으로 홈 팀은 슈팅 수가 많아질수록 유효 슈팅 수가 증가하는 경향이 더 뚜렷하고, 홈 어드밴티지를 통해 더 공격적인 
                        플레이를 펼치며 슈팅이 유효 슈팅으로 연결될 가능성이 크다. 원정 팀은 슈팅 수가 많아질수록 유효 슈팅 수가 증가하지만 
                        홈 팀에 비해 그 영향이 약간 적기 때문에 최대한 슈팅 수를 늘리기 위해 노력해야 한다.
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')

    with tab5:
        st.subheader("홈 팀과 원정 팀의 점유율에 따른 슈팅 수")

        home_correlation = data['home_team_shots'].corr(data['home_team_possession'])
        away_correlation = data['away_team_shots'].corr(data['away_team_possession'])

        X_home = sm.add_constant(data['home_team_possession'])
        model_home = sm.OLS(data['home_team_shots'], X_home).fit()
        home_slope = model_home.params['home_team_possession']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_possession'])
        model_away = sm.OLS(data['away_team_shots'], X_away).fit()
        away_slope = model_away.params['away_team_possession']
        away_intercept = model_away.params['const']

        max_shots = max(data['home_team_shots'].max(), data['away_team_shots'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_possession', y='home_team_shots', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Possession vs Shots\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Possession')
        plt.ylabel('Home Team Shots')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_shots, 1.02 * max_shots)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_possession', y='away_team_shots', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Possession vs Shots\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Possession')
        plt.ylabel('Away Team Shots')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_shots, 1.02 * max_shots)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 점유율이 증가할수록 슈팅 수가 더 많이 증가하는 경향이 있다. 홈 팀이 경기에서 점유율을 높이는 
                        것은 더 많은 슈팅 기회를 창출할 가능성이 크다. Y절편이 2.955인 것은 홈 팀이 낮은 점유율을 유지해도 상대적으로 많은 
                        슈팅 기회를 가질 수 있음을 보여준다. 상관 계수가 0.530으로 중간 정도의 양의 상관 관계가 있다.<br>                        
                        원정 팀의 경우 점유율이 증가할수록 슈팅 수가 증가하지만, 홈 팀에 비해 기울기가 0.114로 낮다. 이는 원정 팀이 
                        점유율을 높이는 것이 슈팅 기회로 직접 연결되기 어려울 수 있음을 의미한다. Y절편이 3.613로 홈 팀보다 높아 원정 팀이 
                        낮은 점유율을 유지하여도 슈팅 기회가 충분히 있음을 보여준다.  상관 계수는 0.473으로 홈 팀보다는 약간 낮다.<br>                        
                        결론적으로 홈 팀은 점유율이 증가할수록 슈팅 수가 더 뚜렷하게 증가하며, 홈 어드밴티지를 통해 더 많은 공격 기회를 
                        창출할 수 있다. 원정 팀은 점유율이 증가할수록 슈팅 수가 증가하지만 그 영향이 홈 팀에 비해 적기 때문에 원정 경기에서 
                        점유율보다는 역습 상황을 노려 슈팅을 때려내는 공격 전술을 사용하는 것이 중요하다.
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')


    with tab6:
        st.subheader("홈 팀과 원정 팀의 점유율에 따른 파울 수")

        home_correlation = data['home_team_fouls'].corr(data['home_team_possession'])
        away_correlation = data['away_team_fouls'].corr(data['away_team_possession'])

        X_home = sm.add_constant(data['home_team_possession'])
        model_home = sm.OLS(data['home_team_fouls'], X_home).fit()
        home_slope = model_home.params['home_team_possession']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_possession'])
        model_away = sm.OLS(data['away_team_fouls'], X_away).fit()
        away_slope = model_away.params['away_team_possession']
        away_intercept = model_away.params['const']

        max_fouls = max(data['home_team_fouls'].max(), data['away_team_fouls'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_possession', y='home_team_fouls', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Possession vs Fouls\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Possession')
        plt.ylabel('Home Team Fouls')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_fouls, 1.02 * max_fouls)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_possession', y='away_team_fouls', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Possession vs Fouls\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Possession')
        plt.ylabel('Away Team Fouls')

        plt.xlim(10, 90)
        plt.ylim(-0.02 * max_fouls, 1.02 * max_fouls)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 점유율이 증가할수록 파울 수가 약간 감소하는 경향이 있다. 홈 팀이 경기에서 점유율을 높이는 것은 
                        파울을 줄이는 데 도움이 될 수 있으며, 이는 더 많은 볼 소유를 통해 경기를 지배할 수 있음을 나타낸다. 상관 계수는 
                        -0.231로 약한 음의 상관 관계가 있다.<br>                        
                        원정 팀의 경우 점유율이 증가할수록 파울 수가 약간 감소하지만, 홈 팀에 비해 기울기가 -0.027로 더 낮다. 이는 원정 
                        팀이 점유율을 높이는 것이 파울 감소로 직접 연결되기 어려울 수 있음을 의미한다. 상관 계수는 -0.105로, 홈 팀보다는 
                        낮지만 여전히 약한 음의 상관 관계를 나타낸다.<br>                        
                        결론적으로 홈 팀은 점유율이 증가할수록 파울 수가 더 뚜렷하게 감소하며, 홈 어드밴티지를 통해 경기의 주도권을 잡고 
                        파울을 줄일 수 있다. 원정 팀은 점유율이 증가할수록 파울 수가 감소하지만 그 영향이 홈 팀에 비해 적기 때문에 원정 
                        경기에서 점유율을 잃지 않기 위해 노력하며 효과적으로 경기해야 한다.<br>
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')

    with tab7:
        st.subheader("홈 팀과 원정 팀의 파울 수에 따른 총 카드 수")

        data['home_team_total_cards'] = data['home_team_yellow_cards'] + data['home_team_red_cards']
        data['away_team_total_cards'] = data['away_team_yellow_cards'] + data['away_team_red_cards']

        home_correlation = data['home_team_total_cards'].corr(data['home_team_fouls'])
        away_correlation = data['away_team_total_cards'].corr(data['away_team_fouls'])

        X_home = sm.add_constant(data['home_team_fouls'])
        model_home = sm.OLS(data['home_team_total_cards'], X_home).fit()
        home_slope = model_home.params['home_team_fouls']
        home_intercept = model_home.params['const']

        X_away = sm.add_constant(data['away_team_fouls'])
        model_away = sm.OLS(data['away_team_total_cards'], X_away).fit()
        away_slope = model_away.params['away_team_fouls']
        away_intercept = model_away.params['const']

        max_fouls = max(data['home_team_fouls'].max(), data['away_team_fouls'].max())
        max_total_cards = max(data['home_team_total_cards'].max(), data['away_team_total_cards'].max())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.regplot(x='home_team_fouls', y='home_team_total_cards', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}, ci=None, truncate=False)
        plt.title(
            f'Home Team Fouls vs Total Cards\n(Slope: {home_slope:.3f}, Y-Intercept: {home_intercept:.3f}, Correlation: {home_correlation:.3f})')
        plt.xlabel('Home Team Fouls')
        plt.ylabel('Home Team Total Cards')

        plt.xlim(-0.02 * max_fouls, 1.02 * max_fouls)
        plt.ylim(-0.02 * max_total_cards, 1.02 * max_total_cards)

        plt.subplot(1, 2, 2)
        sns.regplot(x='away_team_fouls', y='away_team_total_cards', data=data, scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}, ci=None, truncate=False)
        plt.title(
            f'Away Team Fouls vs Total Cards\n(Slope: {away_slope:.3f}, Y-Intercept: {away_intercept:.3f}, Correlation: {away_correlation:.3f})')
        plt.xlabel('Away Team Fouls')
        plt.ylabel('Away Team Total Cards')

        plt.xlim(-0.02 * max_fouls, 1.02 * max_fouls)
        plt.ylim(-0.02 * max_total_cards, 1.02 * max_total_cards)

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                    <div style="font-size: 13px; line-height: 1.8;">
                        홈 팀의 그래프에서는 파울 수가 증가할수록 총 카드 수가 증가하는 경향이 있다. 홈 팀이 경기에서 파울을 
                        많이 할수록 더 많은 카드를 받을 가능성이 크며, 이는 경기에서의 거친 플레이가 더 많은 제재로 이어질 수 있음을 
                        나타낸다. 상관 계수는 0.346으로 중간 정도의 양의 상관 관계가 있다.<br>                        
                        원정 팀의 경우 파울 수가 증가할수록 총 카드 수가 증가하지만, 홈 팀에 비해 기울기가 0.122로 약간 낮다. 
                        이는 원정 팀이 파울을 많이 하더라도 카드를 받을 가능성이 홈 팀보다 약간 낮을 수 있음을 의미한다. 상관 계수는 
                        0.338로, 홈 팀과 비슷한 정도의 양의 상관 관계를 나타낸다.<br>                        
                        결론적으로 홈 팀은 파울 수가 많아질수록 더 많은 카드를 받을 가능성이 있으며, 경기에서의 거친 플레이가 더 많은 제재로 
                        이어질 수 있다. 원정 팀은 파울 수가 증가할수록 카드 수가 증가하지만 그 영향이 홈 팀에 비해 약간 적기 때문에 원정 
                        경기에서 파울을 줄이며 효과적인 경기 운영을 하면 충분한 이득을 볼 수 있다.<br>
                    </div>
                """, unsafe_allow_html=True)

        st.sidebar.write('')


elif info_type == 'Team Stat Analysis':
    st.title('Team Stat Analysis')
    st.sidebar.header('Team Stat Analysis Filters')

    team_name = st.sidebar.selectbox('Team Name', ['Team'] + sorted(matches_data['home_team_name'].unique().tolist()))

    if team_name != 'Team':
        col1, col2 = st.columns([3, 2])

        with col1:
            st.header(team_name)
            st.write(' ')

            team_matches = matches_data[
                (matches_data['home_team_name'] == team_name) | (matches_data['away_team_name'] == team_name)]

            stats = {
                'Goals': (team_matches.apply(
                    lambda x: x['home_team_goal_count'] if x['home_team_name'] == team_name else x[
                        'away_team_goal_count'], axis=1)).mean(),

                'Possession': (team_matches.apply(
                    lambda x: x['home_team_possession'] if x['home_team_name'] == team_name else x[
                        'away_team_possession'], axis=1)).mean(),

                'Shots': (team_matches.apply(
                    lambda x: x['home_team_shots'] if x['home_team_name'] == team_name else x['away_team_shots'],
                    axis=1)).mean(),

                'Shots on Target': (team_matches.apply(
                    lambda x: x['home_team_shots_on_target'] if x['home_team_name'] == team_name else x[
                        'away_team_shots_on_target'], axis=1)).mean(),

                'Corners': (team_matches.apply(
                    lambda x: x['home_team_corner_count'] if x['home_team_name'] == team_name else x[
                        'away_team_corner_count'], axis=1)).mean(),

                'Fouls': (team_matches.apply(
                    lambda x: x['home_team_fouls'] if x['home_team_name'] == team_name else x['away_team_fouls'],
                    axis=1)).mean(),

                'Yellow Cards': (team_matches.apply(
                    lambda x: x['home_team_yellow_cards'] if x['home_team_name'] == team_name else x[
                        'away_team_yellow_cards'], axis=1)).mean(),

                'Red Cards': (team_matches.apply(
                    lambda x: x['home_team_red_cards'] if x['home_team_name'] == team_name else x[
                        'away_team_red_cards'], axis=1)).mean()
            }

            teams = matches_data['home_team_name'].unique()
            league_stats = {stat: [] for stat in stats.keys()}
            team_stats_dict = {}

            for team in teams:
                team_matches = matches_data[
                    (matches_data['home_team_name'] == team) | (matches_data['away_team_name'] == team)]
                team_stats = {}

                for stat in stats.keys():
                    if stat == 'Goals':
                        value = team_matches.apply(
                            lambda x: x['home_team_goal_count'] if x['home_team_name'] == team else x[
                                'away_team_goal_count'], axis=1).mean()

                    elif stat == 'Possession':
                        value = team_matches.apply(
                            lambda x: x['home_team_possession'] if x['home_team_name'] == team else x[
                                'away_team_possession'], axis=1).mean()

                    elif stat == 'Shots':
                        value = team_matches.apply(
                            lambda x: x['home_team_shots'] if x['home_team_name'] == team else x['away_team_shots'],
                            axis=1).mean()

                    elif stat == 'Shots on Target':
                        value = team_matches.apply(
                            lambda x: x['home_team_shots_on_target'] if x['home_team_name'] == team else x[
                                'away_team_shots_on_target'], axis=1).mean()

                    elif stat == 'Corners':
                        value = team_matches.apply(
                            lambda x: x['home_team_corner_count'] if x['home_team_name'] == team else x[
                                'away_team_corner_count'], axis=1).mean()

                    elif stat == 'Fouls':
                        value = team_matches.apply(
                            lambda x: x['home_team_fouls'] if x['home_team_name'] == team else x['away_team_fouls'],
                            axis=1).mean()

                    elif stat == 'Yellow Cards':
                        value = team_matches.apply(
                            lambda x: x['home_team_yellow_cards'] if x['home_team_name'] == team else x[
                                'away_team_yellow_cards'], axis=1).mean()

                    elif stat == 'Red Cards':
                        value = team_matches.apply(
                            lambda x: x['home_team_red_cards'] if x['home_team_name'] == team else x[
                                'away_team_red_cards'], axis=1).mean()

                    league_stats[stat].append(value)
                    team_stats[stat] = value

                team_stats_dict[team] = team_stats

            rankings = {stat: pd.Series(league_stats[stat]).rank(ascending=False, method='min').astype(int).tolist()
                        for stat in stats.keys()}

            team_rankings = {stat: rankings[stat][teams.tolist().index(team_name)] for stat in stats.keys()}

            tactical_features = []

            if stats['Possession'] >= 50:
                tactical_features.append("점유율 우선 전술")

            else:
                tactical_features.append("역습 우선 전술")

            if team_rankings['Shots'] <= 10:
                tactical_features.append("과감한 슈팅 전술")

            else:
                tactical_features.append("신중한 슈팅 전술")

            if team_rankings['Shots on Target'] <= 10:
                tactical_features.append("정확한 슈팅 전술")

            if team_rankings['Corners'] <= 10:
                tactical_features.append("코너킥 유도 전술")

            if team_rankings['Fouls'] <= 10:
                tactical_features.append("거친 플레이 전술")

            else:
                tactical_features.append("안전한 플레이 전술")

            similar_tactical_teams = []

            for team, team_stats in team_stats_dict.items():
                if team == team_name:
                    continue

                team_tactical_features = []

                if team_stats['Possession'] >= 50:
                    team_tactical_features.append("점유율 우선 전술")

                else:
                    team_tactical_features.append("역습 우선 전술")

                if rankings['Shots'][teams.tolist().index(team)] <= 10:
                    team_tactical_features.append("과감한 슈팅 전술")

                else:
                    team_tactical_features.append("신중한 슈팅 전술")

                if rankings['Shots on Target'][teams.tolist().index(team)] <= 10:
                    team_tactical_features.append("정확한 슈팅 전술")

                if rankings['Corners'][teams.tolist().index(team)] <= 10:
                    team_tactical_features.append("코너킥 유도 전술")

                if rankings['Fouls'][teams.tolist().index(team)] <= 10:
                    team_tactical_features.append("거친 플레이 전술")

                else:
                    team_tactical_features.append("안전한 플레이 전술")

                if set(tactical_features) == set(team_tactical_features):
                    similar_tactical_teams.append((team, team_stats))

            advantage_teams = sorted(similar_tactical_teams, key=lambda x: sum(
                rankings[stat][teams.tolist().index(x[0])] for stat in stats.keys()))[:3]

            disadvantage_teams = sorted(similar_tactical_teams, key=lambda x: sum(
                rankings[stat][teams.tolist().index(x[0])] for stat in stats.keys()), reverse=True)[:3]

            if len(advantage_teams) < 3:
                remaining_teams = [team for team in team_stats_dict.keys() if
                                   team not in [x[0] for x in similar_tactical_teams] and team != team_name]
                remaining_teams = sorted(remaining_teams, key=lambda team: sum(team_stats_dict[team].values()))[
                                  :(3 - len(advantage_teams))]
                advantage_teams.extend([(team, team_stats_dict[team]) for team in remaining_teams])

            if len(disadvantage_teams) < 3:
                remaining_teams = [team for team in team_stats_dict.keys() if
                                   team not in [x[0] for x in similar_tactical_teams] and team != team_name]
                remaining_teams = sorted(remaining_teams, key=lambda team: sum(team_stats_dict[team].values()),
                                         reverse=True)[:(3 - len(disadvantage_teams))]
                disadvantage_teams.extend([(team, team_stats_dict[team]) for team in remaining_teams])

            disadvantage_teams = [(team, stats) for team, stats in disadvantage_teams if
                                  team not in [t[0] for t in advantage_teams]]

            if len(disadvantage_teams) < 3:
                remaining_teams = [team for team in team_stats_dict.keys() if
                                   team not in [x[0] for x in similar_tactical_teams] and team not in [x[0] for x in
                                                                                                       advantage_teams] and team != team_name]
                remaining_teams = sorted(remaining_teams, key=lambda team: sum(team_stats_dict[team].values()),
                                         reverse=True)[:(3 - len(disadvantage_teams))]
                disadvantage_teams.extend([(team, team_stats_dict[team]) for team in remaining_teams])

            st.subheader('Team Average Stats')
            st.markdown(f"""
                            <p style="font-size: 20px; line-height: 1.5;">Average Goal: {stats['Goals']:.2f} <strong>(Rank: {team_rankings['Goals']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Possession: {stats['Possession']:.2f} <strong>(Rank: {team_rankings['Possession']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Shot: {stats['Shots']:.2f} <strong>(Rank: {team_rankings['Shots']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Shot on Target: {stats['Shots on Target']:.2f} <strong>(Rank: {team_rankings['Shots on Target']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Corner: {stats['Corners']:.2f} <strong>(Rank: {team_rankings['Corners']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Foul: {stats['Fouls']:.2f} <strong>(Rank: {team_rankings['Fouls']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Yellow Card: {stats['Yellow Cards']:.2f} <strong>(Rank: {team_rankings['Yellow Cards']})</p>
                            <p style="font-size: 20px; line-height: 1.5;">Average Red Card: {stats['Red Cards']:.2f} <strong>(Rank: {team_rankings['Red Cards']})</p>
                        """, unsafe_allow_html=True)

        with col2:
            image_path = f'{team_name}.png'
            st.image(image_path, width=70)

            st.subheader("전술적 특징")
            for feature in tactical_features:
                st.markdown(f"""
                            <p style="font-size: 20px; line-height: 1.1;">{feature}</p>
                        """, unsafe_allow_html=True)

            st.write(' ')


            def handle_click():
                st.write("이미지가 클릭되었습니다!")


            st.subheader("전술상 유리한 팀")
            for team, _ in advantage_teams:
                col11, col22 = st.columns([1, 7])

                with col11:
                    image_path = f'{team}.png'
                    st.image(image_path, width=22)

                with col22:
                    st.markdown(f"""
                                <p style="font-size: 20px; line-height: 1.1;">{team}</p>
                            """, unsafe_allow_html=True)

            st.write(' ')

            st.subheader("전술상 불리한 팀")
            for team, _ in disadvantage_teams:
                col11, col22 = st.columns([1, 7])

                with col11:
                    image_path = f'{team}.png'
                    st.image(image_path, width=22)

                with col22:
                    st.markdown(f"""
                                <p style="font-size: 20px; line-height: 1.1;">{team}</p>
                            """, unsafe_allow_html=True)