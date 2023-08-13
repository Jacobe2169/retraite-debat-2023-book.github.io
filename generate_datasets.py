
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from glob import glob
from typing import Any

import pandas as pd
import spacy
from bs4 import BeautifulSoup, element
from tqdm import tqdm

from lib.utils import cleaner

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


@dataclass
class Intervention:
    section: str
    subsection: str
    text: str
    timestamp: float
    orateur_nom: str
    orateur_slug: str

    def tolist(self) -> list[Any]:
        return [self.section, self.subsection, self.text, self.timestamp, self.orateur_nom, self.orateur_slug]


class SeanceParser:
    def __init__(self, xml_string: str) -> None:
        self.__data: BeautifulSoup = BeautifulSoup(xml_string, features="xml")

    def parse_metadata(self) -> None:
        meta: element.Tag = self.__data.find("metadonnees")

        def get_field(field: str):
            return meta.select_one(field).text

        string_date: str = get_field("dateSeance")
        (year, month, day) = int(string_date[:4]), int(
            string_date[4:6]), int(string_date[6:8])
        self.date_seance: date = date(year, month, day)

        self.num_seance = int(get_field("numSeance"))

        self.legislature: int = int(get_field("legislature"))
        self.presidence: str = get_field("presidentSeance")
        self.presidence_id: str = meta.select_one(
            "presidentSeance").attrs["id_syceron"]

    def parse_intervention(self) -> None:
        self.interventions: list[Intervention] = []
        current_section: str = ""
        current_subsection: str = ""

        for point in self.__data.find_all('point'):
            level: int = int(point.attrs["nivpoint"])
            title: str = point.select_one("texte").text
            if level == 1:
                current_section = title
            elif level == 2:
                current_subsection = title
            for paragraphe in point.find_all("paragraphe"):
                orateur = paragraphe.select_one("orateur")
                inter = paragraphe.select_one("texte")
                if orateur and inter:
                    identifiant: int = int(orateur.id.text)
                    slug = "UNKNOWN"
                    if identifiant in id2slug:
                        slug = id2slug[identifiant]
                    elif identifiant != 0:
                        slug = "GOV"
                    self.interventions.append(Intervention(*[current_section, current_subsection, inter.text,
                                                             eval(
                                                                 inter.attrs["stime"]) if "stime" in inter.attrs else 0,
                                                             orateur.nom.text, slug]))

    def to_dataframe(self) -> pd.DataFrame:
        assert hasattr(self, "interventions")
        assert hasattr(self, "legislature")
        assert hasattr(self, "presidence")
        assert hasattr(self, "presidence_id")
        assert hasattr(self, "num_seance")
        assert hasattr(self, "date_seance")

        data:list[list] = [[self.legislature, self.date_seance, self.num_seance, self.presidence,self.presidence_id, *intervention.tolist()] for intervention in self.interventions]
        return pd.DataFrame(data, columns="legislature date numero_seance nom_president_e id_president_e section subsection full_text timestamp speaker_name slug".split())


def tweet2dataframe(name:str, data:dict):
    try:
        df_extract:pd.DataFrame = pd.DataFrame.from_records([d["legacy"] for d in data if "legacy" in d])
        
        # Date of the tweet
        df_extract["created_at"] = pd.to_datetime(df_extract.created_at)
        
        # Username of the twitter user
        df_extract["username"] = name

        # Retrieve retweet related informations
        df_extract["retweeted_status_result"] = df_extract.retweeted_status_result.apply(
            lambda x: {} if pd.isnull(x) else x)
        df_extract["retweet_id"] = df_extract.retweeted_status_result.apply(
            lambda x: x["result"]["rest_id"] if x else None)
        df_extract["retweet_username"] = df_extract.retweeted_status_result.apply(
            lambda x: x["result"]["core"]["user_results"]["result"]["legacy"]["screen_name"] if x else None)
        df_extract["retweet_user_id"] = df_extract.retweeted_status_result.apply(
            lambda x: x["result"]["core"]["user_results"]["result"]["rest_id"] if x else None)
        
        return df_extract
    except:
        return None


# Load parliamentary data

logging.info("Load parliamentaries's data")
df_deputy:pd.DataFrame = pd.read_csv(
    "data/nosdeputes.fr_deputes_en_mandat_2023-08-02.csv", sep=";")
id2slug:dict = dict(df_deputy["id_an slug".split()].values)

# Load keywords

logging.info("Load keywords dataset")
keyword_df:pd.DataFrame = pd.read_excel("data/keywords_selected.ods", index_col=0)
keyword_df.drop_duplicates("lemmatized", inplace=True)


# Build twitter dataset
logging.info("Load and parse twitter extractions")
columns:list = "username full_text created_at in_reply_to_screen_name in_reply_to_status_id_str in_reply_to_user_id_str retweet_id retweet_username retweet_user_id is_quote_status quoted_status_id_str".split()
twitter_extraction_files:list[str] = ["data/raw_data/extract_until_fev_23_part1.json","data/raw_data/extract_until_fev_23_part2.json"]

# Read and parse twitter extraction files
dfs:list[pd.DataFrame] = []
for ix,filename in enumerate(twitter_extraction_files):
    data:dict = json.load(open(filename))
    data = {v["username_key"]: v["tweets"]
            for _, v in data["_default"].items() if "tweets" in v}
    df:pd.DataFrame = pd.concat([tweet2dataframe(slug, data_i) 
            for slug, data_i in tqdm(data.items(), desc=f"Parse twitter extraction... n°{ix+1}")])[columns]
    dfs.append(df)

twitter_df:pd.DataFrame = pd.concat(dfs).sort_values(by="created_at")

logging.info("Clean tweets, get groupe per parliamentary, identify hashtag")
# Filter out tweets not in the designated period
twitter_df_fev_juin:pd.DataFrame = twitter_df[(twitter_df["created_at"] < "2023-06-08") & (twitter_df["created_at"] > "2023-02-01")].copy()

# Assign a parliamentary group for each twitter user
twitter_df_fev_juin["groupe_sigle"] = twitter_df_fev_juin.username.map(dict(df_deputy["slug groupe_sigle".split()].values))

# Extract Hashtags
pattern_hashtag = re.compile(r"#\b\w\w+\b")
twitter_df_fev_juin["hashtag"] = twitter_df_fev_juin.full_text.apply(lambda x: pattern_hashtag.findall(x))
twitter_df_fev_juin["is_hashtag"] = twitter_df_fev_juin.hashtag.apply(lambda x: len(x) > 0)

# Clean tweets
twitter_df_fev_juin["full_text"] = twitter_df_fev_juin.full_text.apply(cleaner)
twitter_df_fev_juin.rename(columns=dict(created_at="date"), inplace=True)

# Build assemblée nationale reports dataset
logging.info("Load and parse report from the assemblée nationale")
dataframes:list[pd.DataFrame] = []
for filename in tqdm(glob("data/raw_data/compteRendu/*.xml"), desc="Parse CompteRendu xml files"):
    xml_str = open(filename, encoding="utf-8").read()
    s = SeanceParser(xml_str)
    s.parse_metadata()
    s.parse_intervention()
    dataframes.append(s.to_dataframe())

parlement_df:pd.DataFrame = pd.concat(dataframes)
parlement_df.rename(columns=dict(slug="username"), inplace=True)

# Filter out non related to the pension reform
logging.info("Filter out intervention unrelated to the french pension reform of 2023")
section_name:str = "Projet de loi de financement rectificative de la sécurité sociale pour 2023"
parlement_df = parlement_df[parlement_df.section == section_name]

# Assign a group for each intervention speaker
logging.info("assign parliamentary group for each intervention's speaker")
slug2groupe:dict = dict(df_deputy["slug groupe_sigle".split()].values)

def getSigleGroupeParlementaire(row:pd.Series)->str:
    if row.username in ["GOV", "UNKNOWN"]:
        return row.username
    return slug2groupe[row.username]

parlement_df["groupe_sigle"] = parlement_df.apply(getSigleGroupeParlementaire, axis=1)

# Load spacy model for lemmatization
logging.info("Load Spacy model")
nlp:spacy.Language = spacy.load("fr_core_news_md", disable=("ner", "textcat", "parser"))

# Lemmatize and detect keywords
logging.info("Lemmatize and detect keywords in both datasets")
for df in [twitter_df_fev_juin, parlement_df]:
    df["lemmatization"] = [" ".join([token.lemma_ for token in doc]) 
                            for doc in tqdm(nlp.pipe(df.full_text.values), total=len(df))]
    df["keywords_detected"] = df.lemmatization.apply(
        lambda x: [str(keyword) for keyword in keyword_df.lemmatized if str(keyword) in x])
    df["is_keywords"] = df.keywords_detected.apply(
        lambda x: len(x) > 0)


# Save Data
logging.info("Saving data")
parlement_df.to_parquet("parlement_retraite_data.parquet")
twitter_df_fev_juin.to_parquet("twitter_fev_to_juin_2023_retraite_data.parquet")
