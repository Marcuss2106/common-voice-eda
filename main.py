import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(15, 6))

TRAIN_FILE_PATH = "./data/train.tsv"

assert os.path.exists(TRAIN_FILE_PATH), f"File not found: {TRAIN_FILE_PATH}"


def main():
    print("Hello from audio-explorer!")
    df = load_data(TRAIN_FILE_PATH)
    print(df.head())
    drop_columns(df)
    unique_accents_df = unique_accents(df)
    print("\nUnique Accents:", unique_accents_df.size)
    print("\n")
    samples_by_gender_df = samples_by_gender(df)
    print(samples_by_gender_df)
    samples_by_accent(df)
    dialect_distribution(df)
    sentence_length(df)
    samples_by_age_group(df)
    finalize_plot()


def unique_accents(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        print("DataFrame is empty.")
        return pd.Series()

    unique_accents = df["accents"].unique()
    unique_accents = pd.Series(unique_accents).dropna()
    return unique_accents


def samples_by_gender(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        print("DataFrame is empty.")
        return pd.Series()

    df["gender"].dropna(inplace=True)
    return df["gender"].value_counts()


def samples_by_accent(df: pd.DataFrame):
    if df.empty:
        print("DataFrame is empty.")
        return df

    df = df.groupby("accents", dropna=True).size().reset_index()
    df.columns = ["accents", "count"]
    df = df.sort_values(by="count", ascending=False)
    df = df.iloc[0:10]
    sns.barplot(x="count", y="accents", data=df, palette="tab10", hue="accents", ax=ax1)
    ax1.set_title("Unique Accents Count")
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Accents")


def dialect_distribution(df: pd.DataFrame):
    if df.empty:
        print("DataFrame is empty.")
        return df
    df = df.groupby("accents", dropna=True).size().reset_index()
    df.columns = ["accents", "count"]
    df = df.sort_values(by="count", ascending=False)
    df = df.iloc[0:10]

    percents = 100.0 * df["count"] / df["count"].sum()
    colors, text = ax2.pie(df["count"], startangle=90)
    labels = [
        "{0} — {1:1.2f}%".format(accent, count)
        for accent, count in zip(df["accents"], percents)
    ]
    ax2.legend(colors, labels, loc="center left", bbox_to_anchor=(-2, 0.5), fontsize=8)
    ax2.set_title("Dialect Distribution")


def sentence_length(df: pd.DataFrame):
    if df.empty:
        print("DataFrame is empty.")
        return df

    sentence_lengths = df["sentence"].apply(lambda x: len(x.split()))
    sentence_lengths = sentence_lengths[sentence_lengths < 16]
    counts = sentence_lengths.value_counts().sort_index()
    mean_length = sentence_lengths.mean()
    smoothed_counts = counts.rolling(window=3, center=True, min_periods=1).mean()
    ax3.bar(counts.index, counts.values, width=0.9, alpha=0.5, label="Histogram")
    ax3.plot(
        smoothed_counts.index, smoothed_counts.values, color="b", label="Smoothed Line"
    )
    ax3.axvline(x=mean_length, color="r", label=f"Mean: ≈{round(mean_length)} words")
    ax3.legend()
    ax3.set_title("Sentence Length Distribution")
    ax3.set_xlabel("Length of Sentence in Words")
    ax3.set_ylabel("Frequency")


def samples_by_age_group(df: pd.DataFrame):
    if df.empty:
        print("DataFrame is empty.")
        return df

    samples_by_age_group = df["age"].dropna().value_counts()
    sns.countplot(
        x="age",
        data=df,
        order=samples_by_age_group.index,
        palette="Spectral",
        hue="age",
        ax=ax4,
        legend=False,
    )
    ax4.set_title("Samples by Age Group")
    ax4.tick_params(axis="x", labelrotation=45)


def finalize_plot():
    sns.despine()
    plt.tight_layout()
    plt.show()


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path, sep="\t")
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("DataFrame is empty.")
        return df

    required_columns = ["sentence", "age", "gender", "accents"]
    drop_columns = [col for col in df.columns if col not in required_columns]
    df.drop(columns=drop_columns, axis=1, inplace=True)

    print("Unused columns dropped successfully")

    return df


if __name__ == "__main__":
    main()
