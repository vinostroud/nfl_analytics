# analytics_py.py

import pandas as pd
import nfl_data_py as nfl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(df_year: int):
    df = nfl.import_pbp_data([df_year], downcast=True)
    df = df[df["season_type"] == "REG"]
    df = df[(df["pass"] == 1) | (df["rush"] == 1)]
    df = df.dropna(subset=["epa", "posteam", "defteam"])
    return df


def get_mean_epa_down1(df):
    return (
        df[df["down"] == 1]
        .groupby("posteam")["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "Team", "epa": "First Down EPA"})
        .assign(Rank=lambda x: x["First Down EPA"].rank(ascending=False, method="min"))
        .sort_values(by="First Down EPA", ascending=False)
        .reset_index(drop=True)
    )


def get_mean_epa_down1and2(df):
    return (
        df[df["down"].isin([1, 2])]
        .groupby("posteam")["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "Team", "epa": "EPA Downs One and Two"})
        .assign(
            Rank=lambda x: x["EPA Downs One and Two"].rank(
                ascending=False, method="min"
            )
        )
        .sort_values(by="EPA Downs One and Two", ascending=False)
        .reset_index(drop=True)
    )


def get_game_by_game_data(df):
    grouped_df_home = (
        df.groupby(["game_id", "home_team", "home_score", "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .reset_index()
        .rename(
            columns={
                "game_id": "Game ID",
                "home_team": "Home Team",
                "epa": "Home Team EPA",
                "home_score": "Home Team Score",
            }
        )
    )

    filtered_df_home = grouped_df_home.assign(
        **{"Home Team turnovers": lambda x: x["fumble_lost"] + x["interception"]}
    )[grouped_df_home["Home Team"] == grouped_df_home["posteam"]][
        [
            "Game ID",
            "Home Team",
            "Home Team EPA",
            "Home Team turnovers",
            "Home Team Score",
        ]
    ].reset_index(
        drop=True
    )

    grouped_df_away = (
        df.groupby(["game_id", "away_team", "away_score", "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .reset_index()
        .rename(
            columns={
                "game_id": "Game ID",
                "away_team": "Away Team",
                "epa": "Away Team EPA",
                "away_score": "Away Team Score",
            }
        )
        .assign(
            **{"Away Team turnovers": lambda x: x["fumble_lost"] + x["interception"]}
        )
    )

    filtered_df_away = grouped_df_away[
        grouped_df_away["Away Team"] == grouped_df_away["posteam"]
    ][
        [
            "Game ID",
            "Away Team",
            "Away Team EPA",
            "Away Team turnovers",
            "Away Team Score",
        ]
    ].reset_index(
        drop=True
    )

    # merged_df_extended = pd.merge(filtered_df_home, filtered_df_away, on='Game ID')

    df_consolidated_epa_score_away = filtered_df_away.rename(
        columns={
            "game_id": "Game ID",
            "Away Team": "Team",
            "Away Team EPA": "EPA",
            "Away Team turnovers": "Turnovers",
            "Away Team Score": "Score",
        }
    ).reset_index(drop=True)

    df_consolidated_epa_score_home = filtered_df_home.rename(
        columns={
            "game_id": "Game ID",
            "Home Team": "Team",
            "Home Team EPA": "EPA",
            "Home Team turnovers": "Turnovers",
            "Home Team Score": "Score",
        }
    ).reset_index(drop=True)

    df_consolidated_epa_score_combined = pd.concat(
        [df_consolidated_epa_score_away, df_consolidated_epa_score_home], axis=0
    ).reset_index(drop=True)

    return df_consolidated_epa_score_combined


def process_team_data(df, team_col, score_col):
    grouped_df = (
        df.groupby(["game_id", team_col, score_col, "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .reset_index()
        .rename(
            columns={
                "game_id": "Game ID",
                team_col: "Team",
                "epa": "EPA",
                score_col: "Score",
            }
        )
    )

    grouped_df["Turnovers"] = grouped_df["fumble_lost"] + grouped_df["interception"]
    filtered_df = grouped_df[grouped_df["Team"] == grouped_df["posteam"]].copy()
    filtered_df = filtered_df[["Game ID", "Team", "EPA", "Turnovers", "Score"]]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def prepare_data(df):
    home_data = process_team_data(df, "home_team", "home_score")
    away_data = process_team_data(df, "away_team", "away_score")

    df_consolidated = pd.concat([home_data, away_data], axis=0).reset_index(drop=True)
    return df_consolidated


def train_and_plot_regression(df_consolidated):
    # Extract the features and target variable
    x_epa = df_consolidated["EPA"].values.reshape(-1, 1)
    y_score = df_consolidated["Score"].values

    # Split the data into training and testing sets
    x_epa_train, x_epa_test, y_score_train, y_score_test = train_test_split(
        x_epa, y_score, test_size=0.2, random_state=42
    )

    reg = LinearRegression()
    reg.fit(x_epa_train, y_score_train)

    score_y_pred = reg.predict(x_epa_test)

    sfig, ax = plt.subplots()
    ax.scatter(x_epa_test, y_score_test, color="black", label="Actual")
    ax.plot(x_epa_test, score_y_pred, color="blue", linewidth=3, label="Predicted")

    ax.set_xlabel("EPA")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45)
    ax.legend()

    return sfig
