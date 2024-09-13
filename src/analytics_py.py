# analytics_py.py

import pandas as pd
import nfl_data_py as nfl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import dask.dataframe as dd


def load_data(df_year: int):
    ddf = dd.from_pandas(nfl.import_pbp_data([df_year], downcast=True), npartitions=10)
    ddf = ddf[ddf["season_type"] == "REG"]
    ddf = ddf[(ddf["pass"] == 1) | (ddf["rush"] == 1)]
    ddf = ddf.dropna(subset=["epa", "posteam", "defteam"])

    return ddf


def get_mean_epa_down1(ddf):
    mean_epa_down1 = ddf[ddf["down"] == 1].groupby("posteam")["epa"].mean().compute()
    return mean_epa_down1.reset_index().rename(
        columns={"posteam": "Team", "epa": "First Down EPA"}
    )


def get_mean_epa_down1and2(ddf):
    mean_epa_down1and2 = (
        ddf[ddf["down"].isin([1, 2])].groupby("posteam")["epa"].mean().compute()
    )
    return mean_epa_down1and2.reset_index().rename(
        columns={"posteam": "Team", "epa": "EPA Downs One and Two"}
    )


def get_game_by_game_data(ddf):
    # Process home team data
    grouped_df_home = (
        ddf.groupby(["game_id", "home_team", "home_score", "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .reset_index()
        .compute()
    )

    filtered_df_home = (
        grouped_df_home.assign(
            **{"Home Team turnovers": lambda x: x["fumble_lost"] + x["interception"]}
        )[grouped_df_home["home_team"] == grouped_df_home["posteam"]][
            ["home_team", "epa", "fumble_lost", "interception", "home_score"]
        ]
        .rename(
            columns={
                "home_team": "Team",
                "epa": "Home Team EPA",
                "home_score": "Home Team Score",
            }
        )
        .reset_index(drop=True)
    )

    # Process away team data
    grouped_df_away = (
        ddf.groupby(["game_id", "away_team", "away_score", "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .reset_index()
        .compute()
    )

    filtered_df_away = (
        grouped_df_away.assign(
            **{"Away Team turnovers": lambda x: x["fumble_lost"] + x["interception"]}
        )[grouped_df_away["away_team"] == grouped_df_away["posteam"]][
            ["away_team", "epa", "fumble_lost", "interception", "away_score"]
        ]
        .rename(
            columns={
                "away_team": "Team",
                "epa": "Away Team EPA",
                "away_score": "Away Team Score",
            }
        )
        .reset_index(drop=True)
    )

    # Combine home and away data using merge to ensure alignment by team
    df_consolidated_epa_score_combined = pd.merge(
        filtered_df_home,
        filtered_df_away,
        on="Team",
        how="outer",
        suffixes=("_home", "_away"),
    )

    return df_consolidated_epa_score_combined


def process_team_data(ddf, team_col, score_col):
    grouped_df = (
        ddf.groupby(["game_id", team_col, score_col, "posteam"])
        .agg({"epa": "mean", "fumble_lost": "sum", "interception": "sum"})
        .compute()
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


def prepare_data(ddf):
    home_data = process_team_data(ddf, "home_team", "home_score")
    away_data = process_team_data(ddf, "away_team", "away_score")

    df_consolidated = (
        dd.concat([home_data, away_data], axis=0).compute().reset_index(drop=True)
    )
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
