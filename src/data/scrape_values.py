import datetime as dt
import re
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from lxml import html as LH

FROM_DATE = "2015-07-01"
TO_DATE   = "2025-06-30"

# Base output for valuations
OUTPUT_CSV = "data/raw/tm_pl_all_columns.csv"

COMPETITIONS = {
    "GB1": "https://www.transfermarkt.com/premier-league/marktwerteverein/wettbewerb/GB1",
    "GB2": "https://www.transfermarkt.com/championship/marktwerteverein/wettbewerb/GB2",
    "GB3": "https://www.transfermarkt.com/league-one/marktwerteverein/wettbewerb/GB3",
}

# Expected valuation columns coming out of table parsing
EXPECTED_VAL_COLS = [
    "rank", "team", "league_at_date", "value_eur_at_date", "squad_size_at_date",
    "current_value_eur", "current_squad_size", "difference_eur", "difference_pct"
]

def _mk_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1.2, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

def _semi_monthly_dates(start_iso, end_iso):
    start = dt.date.fromisoformat(start_iso)
    end = dt.date.fromisoformat(end_iso)
    cur = dt.date(start.year, start.month, 1)
    out = []
    while cur <= end:
        d1 = dt.date(cur.year, cur.month, 1)
        d2 = dt.date(cur.year, cur.month, 15)
        if start <= d1 <= end:
            out.append(d1)
        if start <= d2 <= end:
            out.append(d2)
        cur = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return sorted(set(out))

_VAL_RE = re.compile(r"([€$£])?\s*([\-]?\d[\d\.,]*)\s*(bn|m|th|Th\.)?", re.I)

def _money_to_eur(s):
    if not isinstance(s, str):
        return None
    s = s.replace("\xa0", " ").strip()
    if not s or s == "-":
        return None
    m = _VAL_RE.search(s)
    if not m:
        return None
    _, num, unit = m.groups()
    num = num.replace(",", "")
    try:
        val = float(num)
    except ValueError:
        return None
    u = (unit or "").lower()
    if u == "bn":
        val *= 1e9
    elif u == "m":
        val *= 1e6
    elif u in ("th", "th."):
        val *= 1e3
    return val

def _pct(s):
    try:
        return float(str(s).replace("%", "").replace("\xa0", " ").strip())
    except Exception:
        return None

def _thead_texts(table):
    ths = table.xpath('.//thead//th')
    return [" ".join(t.xpath(".//text()")).strip() for t in ths]

def _infer_column_indices(table):
    """
    Map logical columns -> td indices by reading THEAD headers.
    """
    headers = _thead_texts(table)
    norm = [re.sub(r"\s+", " ", h) for h in headers]
    want = {
        "rank":        re.compile(r"^\s*#\s*$|no\.", re.I),
        "club":        re.compile(r"\bclub\b", re.I),
        "league":      re.compile(r"\bleague\b.*", re.I),
        "value":       re.compile(r"\bvalue\b(?!.*current)", re.I),
        "size":        re.compile(r"\bsquad\s*size\b", re.I),
        "cur_value":   re.compile(r"\bcurrent\s*value\b", re.I),
        "cur_size":    re.compile(r"\bcurrent\s*squad\s*size\b", re.I),
        "difference":  re.compile(r"^\s*difference\s*(\(€\))?\s*$", re.I),
        "percent":     re.compile(r"^\s*%(\s|$)|percent", re.I),
    }
    index_map = {}
    for idx, h in enumerate(norm):
        for key, rx in want.items():
            if key in index_map:
                continue
            if rx.search(h):
                if key == "value" and re.search(r"current\s*value", h, re.I):
                    continue
                index_map[key] = idx
    return index_map

def _extract_rows_from_items_table(table):
    """
    Extract rows from one <table class='items'> using header-mapped TD positions.
    Returns a DataFrame with all expected columns; may be empty.
    """
    idx = _infer_column_indices(table)
    rows = table.xpath('./tbody/tr')
    recs = []
    for r in rows:
        tds = r.xpath('./td')
        if not tds:
            continue

        def get_td(i: int) -> str:
            if i < 0 or i >= len(tds):
                return ""
            return tds[i].text_content().strip()

        club_i = idx.get("club", 2)
        team_txts = tds[club_i].xpath('.//a/text()') or [get_td(club_i)]
        team = " ".join(t.strip() for t in team_txts if t and t.strip()).strip()
        if not team:
            continue

        rec = {
            "rank":                pd.to_numeric(get_td(idx.get("rank", 0)), errors="coerce"),
            "team":                team,
            "league_at_date":      get_td(idx.get("league", 3)),
            "value_eur_at_date":   _money_to_eur(get_td(idx.get("value", 4))),
            "squad_size_at_date":  pd.to_numeric(get_td(idx.get("size", 5)), errors="coerce"),
            "current_value_eur":   _money_to_eur(get_td(idx.get("cur_value", 6))),
            "current_squad_size":  pd.to_numeric(get_td(idx.get("cur_size", 7)), errors="coerce"),
            "difference_eur":      _money_to_eur(get_td(idx.get("difference", 8))),
            "difference_pct":      _pct(get_td(idx.get("percent", 9))),
        }
        recs.append(rec)

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df
    for c in ("rank", "squad_size_at_date", "current_squad_size"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def _score_table(df):
    """Heuristic score to pick the correct club-value table."""
    if df is None or df.empty:
        return -10
    n = len(df)
    has_cols = sum([
        int("current_value_eur" in df and df["current_value_eur"].notna().any()),
        int("difference_eur" in df and df["difference_eur"].notna().any()),
        int("squad_size_at_date" in df and df["squad_size_at_date"].notna().any()),
    ])
    money_rows = int(df["value_eur_at_date"].notna().sum()) + int(df["current_value_eur"].notna().sum())
    shape_bonus = 2 if 18 <= n <= 26 else 0
    return 3 * money_rows + 2 * has_cols + shape_bonus

def _best_items_table_df(html_text):
    """Pick the single best <table class='items'> from a page."""
    doc = LH.fromstring(html_text)
    tables = doc.xpath('//table[contains(@class,"items")]')
    best_df = None
    best_score = -10
    for t in tables:
        try:
            df_t = _extract_rows_from_items_table(t)
        except Exception:
            continue
        sc = _score_table(df_t)
        if sc > best_score:
            best_df, best_score = df_t, sc
    if best_df is None or best_df.empty:
        return pd.DataFrame(), -10
    return best_df, best_score

def _postprocess_valuation_df(df_raw, cutoff):
    """
    Ensure dated value exists, fill from current - difference if needed,
    enforce expected columns and add cutoff_date / competition_code.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["cutoff_date", "competition_code"] + EXPECTED_VAL_COLS)

    if "value_eur_at_date" not in df_raw.columns:
        df_raw["value_eur_at_date"] = pd.NA

    if (df_raw["value_eur_at_date"].isna().any()and df_raw["current_value_eur"].notna().any() and df_raw["difference_eur"].notna().any()):
        df_raw["value_eur_at_date"] = df_raw["value_eur_at_date"].where(
            df_raw["value_eur_at_date"].notna(),
            df_raw["current_value_eur"] - df_raw["difference_eur"],
        )

    for c in EXPECTED_VAL_COLS:
        if c not in df_raw.columns:
            df_raw[c] = pd.NA

    df = df_raw[EXPECTED_VAL_COLS].copy()
    df.insert(0, "cutoff_date", pd.to_datetime(cutoff))
    df.insert(1, "competition_code", pd.NA)
    return df

def normalize_team_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()

    name_map = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Man Utd": "Manchester United",
        "Nott'm Forest": "Nottingham Forest",
        "Nottm Forest": "Nottingham Forest",
        "Wolves": "Wolverhampton Wanderers",
        "Brighton": "Brighton & Hove Albion",
        "Tottenham": "Tottenham Hotspur",
        "Spurs": "Tottenham Hotspur",
        "Newcastle": "Newcastle United",
        "West Ham": "West Ham United",
        "Leicester": "Leicester City",
        "Ipswich": "Ipswich Town",
        "Luton": "Luton Town",
        "Sheffield Utd": "Sheffield United",
        "Sheff Utd": "Sheffield United",
        "Sheffield U.": "Sheffield United",
        "Southampton": "Southampton FC",
        "Cardiff": "Cardiff City",
        "Huddersfield": "Huddersfield Town",
        "Hull": "Hull City",
        "Middlesbrough": "Middlesbrough FC",
        "Norwich": "Norwich City",
        "Stoke": "Stoke City",
        "Swansea": "Swansea City",
        "Watford": "Watford FC",
        "West Brom": "West Bromwich Albion",
        "Leeds": "Leeds United",
        "Arsenal FC": "Arsenal",
        "AFC Bournemouth": "Bournemouth",
        "Brentford FC": "Brentford",
        "Brighton & Hove Albion": "Brighton & Hove Albion",
        "Burnley FC": "Burnley",
        "Chelsea FC": "Chelsea",
        "Everton FC": "Everton",
        "Fulham FC": "Fulham",
        "Liverpool FC": "Liverpool",
        "Sunderland AFC": "Sunderland",
    }

    return name_map.get(name, name)

def analyze_valuation_coverage(processed_csv ="data/processed/processed_15_25_n10.csv", after_date = None, valuation_csv=OUTPUT_CSV):
    try:
        df = pd.read_csv(processed_csv)
        print(f"Loaded {len(df)} matches from {processed_csv}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if after_date:
            cutoff = pd.to_datetime(after_date, errors="coerce")
            df = df[df["date"] >= cutoff]
            print(f"Filtered to {len(df)} matches after {after_date} using 'date'")

        val_cols = [
            "home_squad_value_log_z",
            "away_squad_value_log_z",
            "squad_value_log_advantage_z",
        ]
        missing_cols = [col for col in val_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Run scraper/merge to add valuation data.")
            return {"error": f"Missing columns: {missing_cols}"}

        total_rows = len(df)
        stats = {}
        for col in val_cols:
            null_count = df[col].isna().sum()
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            non_null_count = total_rows - null_count
            coverage_pct = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
            stats[col] = {
                "total": total_rows,
                "null_count": null_count,
                "non_null_count": non_null_count,
                "null_percentage": null_pct,
                "coverage_percentage": coverage_pct,
            }

        print("\n" + "="*70)
        print("VALUATION DATA COVERAGE ANALYSIS")
        print("="*70)
        print(f"Total matches analyzed: {total_rows}")
        if after_date:
            print(f"Date filter: After {after_date}")
        print("\n" + "-"*70)
        for col in val_cols:
            s = stats[col]
            print(f"\n{col}:")
            print(f"  Coverage: {s['non_null_count']}/{s['total']} matches ({s['coverage_percentage']:.2f}%)")
            print(f"  Missing:  {s['null_count']}/{s['total']} matches ({s['null_percentage']:.2f}%)")

        both_present = df[["home_squad_value_log_z", "away_squad_value_log_z"]].notna().all(axis=1).sum()
        both_pct = (both_present / total_rows) * 100 if total_rows > 0 else 0
        print("\n" + "-"*70)
        print("\nComplete match coverage (both teams have data):")
        print(f"  {both_present}/{total_rows} matches ({both_pct:.2f}%)")

        if "date" in df.columns and "home_squad_value_log_z" in df:
            notna_mask = df["home_squad_value_log_z"].notna()
            if notna_mask.any():
                first_val = df.loc[notna_mask, "date"].min()
                last_val = df.loc[notna_mask, "date"].max()
                try:
                    print(f"\nValuation data date range: {first_val.date()} to {last_val.date()}")
                except Exception:
                    print(f"\nValuation data date range: {first_val} to {last_val}")

        print("\n" + "-"*70)
        print("TEAM-LEVEL COVERAGE")
        print("-"*70)
        home_teams = set(df["home_team"].unique()) if "home_team" in df.columns else set()
        away_teams = set(df["away_team"].unique()) if "away_team" in df.columns else set()
        all_teams = sorted(home_teams | away_teams)
        print(f"\nTotal unique teams in match data: {len(all_teams)}")

        teams_with_home_data = set(df[df.get("home_squad_value_log_z").notna()]["home_team"].unique()) if "home_squad_value_log_z" in df else set()
        teams_with_away_data = set(df[df.get("away_squad_value_log_z").notna()]["away_team"].unique()) if "away_squad_value_log_z" in df else set()
        teams_with_any_data = teams_with_home_data | teams_with_away_data
        print(f"Teams with valuation data: {len(teams_with_any_data)}")
        print(f"Teams without valuation data: {len(all_teams) - len(teams_with_any_data)}")

        teams_without_data = sorted(set(all_teams) - teams_with_any_data)
        if teams_without_data:
            print(f"\nTeams missing valuation data ({len(teams_without_data)}):")
            for i, team in enumerate(teams_without_data, 1):
                print(f"  {i}. {team}")

        if teams_with_any_data:
            print(f"\nTeams with valuation data ({len(teams_with_any_data)}):")
            teams_with_data_list = sorted(teams_with_any_data)
            for i, team in enumerate(teams_with_data_list, 1):
                if "home_team" in df.columns and "away_team" in df.columns:
                    match_count = len(df[(df["home_team"] == team) | (df["away_team"] == team)])
                    covered_count = len(
                        df[
                            ((df["home_team"] == team) & df["home_squad_value_log_z"].notna())
                            | ((df["away_team"] == team) & df["away_squad_value_log_z"].notna())
                        ]
                    )
                else:
                    match_count = 0
                    covered_count = 0
                print(f"  {i}. {team}: {covered_count}/{match_count} matches")

        try:
            val_data = pd.read_csv(valuation_csv)
            scraped_teams = sorted(val_data["team"].unique())

            print("\n" + "-"*70)
            print("TEAM NAME COMPARISON")
            print("-"*70)
            print(f"\nTeams in scraped valuation data: {len(scraped_teams)}")

            def _norm_map(names):
                return {normalize_team_name(t): t for t in names}

            match_teams_normalized = _norm_map(all_teams)
            scraped_teams_normalized = _norm_map(scraped_teams)
            match_norm_set = set(match_teams_normalized.keys())
            scraped_norm_set = set(scraped_teams_normalized.keys())
            in_match_not_scraped_norm = sorted(match_norm_set - scraped_norm_set)
            in_scraped_not_match_norm = sorted(scraped_norm_set - match_norm_set)
            print("\nAfter normalization:")
            print(f"  Matched teams: {len(match_norm_set & scraped_norm_set)}")
            print(f"  Unmatched in match data: {len(in_match_not_scraped_norm)}")
            print(f"  Unmatched in valuation data: {len(in_scraped_not_match_norm)}")
            if in_match_not_scraped_norm:
                print("\nTeams in match data but NOT in valuation data (after normalization):")
                for norm_team in in_match_not_scraped_norm:
                    orig = match_teams_normalized[norm_team]
                    print(f"  - {orig} (normalized: {norm_team})")
            if in_scraped_not_match_norm:
                print("\nTeams in valuation data but NOT in match data (after normalization):")
                for norm_team in in_scraped_not_match_norm:
                    orig = scraped_teams_normalized[norm_team]
                    print(f"  - {orig} (normalized: {norm_team})")

        except FileNotFoundError:
            print(f"\nWarning: Could not load {valuation_csv} for team name comparison")
        except Exception as e:
            print(f"\nError during team name comparison: {e}")

        print("="*70 + "\n")

        stats["overall"] = {
            "total_matches": total_rows,
            "complete_matches": both_present,
            "complete_percentage": both_pct,
        }
        return stats

    except FileNotFoundError:
        print(f"Error: Could not find {processed_csv}")
        return {"error": f"File not found: {processed_csv}"}
    except Exception as e:
        print(f"Error analyzing coverage: {e}")
        return {"error": str(e)}


def main():
    dates = _semi_monthly_dates(FROM_DATE, TO_DATE)

    sess = _mk_session()
    frames = []

    for i, d in enumerate(dates, 1):
        total_rows_for_date = 0
        for comp_code, base in COMPETITIONS.items():
            try:
                url = f"{base}/stichtag/{d:%Y-%m-%d}/plus/1"
                r = sess.get(url, timeout=30)
                r.raise_for_status()
                df_raw, score = _best_items_table_df(r.text)
                if df_raw is None or df_raw.empty:
                    continue

                df = _postprocess_valuation_df(df_raw, d)
                if df.empty:
                    continue

                frames.append(df)
                total_rows_for_date += len(df)

            except Exception as e:
                print(f"[{i}/{len(dates)}] {d} [{comp_code}]: ERROR: {e}")

        print(f"[{i}/{len(dates)}] {d} -> {total_rows_for_date} rows across competitions")
        time.sleep(1.25)

    out = (
        pd.concat(frames, ignore_index=True)
        .assign(
            team=lambda x: (
                x["team"]
                .str.replace(r"\s+\(\d+\)$", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        )
        .sort_values(["cutoff_date", "competition_code", "rank", "team"])
        .reset_index(drop=True)
    )

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_CSV}")
    
    processed_dir = Path("data/processed")
    csv_files = sorted(processed_dir.glob("*.csv"))
    
    val_df = out[["cutoff_date", "team", "value_eur_at_date", "squad_size_at_date"]].copy()
    val_df["team_normalized"] = val_df["team"].apply(normalize_team_name)
    val_df = val_df.sort_values(["team_normalized", "cutoff_date"])

    def merge_team_values(match_df, val_df, team_col, team_col_normalized, date_col_for_merge, prefix):
        merged = match_df.copy()
        value_col = f"{prefix}_squad_value"
        if value_col not in merged:
            merged[value_col] = None

        for idx, row in merged.iterrows():
            team_normalized = row[team_col_normalized]
            match_date = row[date_col_for_merge]
            team_vals = val_df[(val_df["team_normalized"] == team_normalized) & (val_df["cutoff_date"] <= match_date)]
            if not team_vals.empty:
                latest = team_vals.iloc[-1]
                merged.at[idx, value_col] = latest["value_eur_at_date"]
        return merged

    # Merge valuations into all processed CSV files
    for processed_file in csv_files:
        print(f"\n{'='*70}")
        print(f"Processing: {processed_file.name}")
        print(f"{'='*70}")
        
        try:
            match_df = pd.read_csv(processed_file)
            print(f"Loaded match data: {len(match_df)} rows")
            
            match_df["_match_date"] = pd.to_datetime(match_df["date"], errors="coerce")
            
            print("Merging home team valuations...")
            match_df["home_team_normalized"] = match_df["home_team"].apply(normalize_team_name) if "home_team" in match_df.columns else None
            match_df = merge_team_values(match_df, val_df, "home_team", "home_team_normalized", "_match_date", "home")
            
            print("Merging away team valuations...")
            match_df["away_team_normalized"] = match_df["away_team"].apply(normalize_team_name) if "away_team" in match_df.columns else None
            match_df = merge_team_values(match_df, val_df, "away_team", "away_team_normalized", "_match_date", "away")

            if "home_squad_value" in match_df.columns and "away_squad_value" in match_df.columns:
                match_df["home_squad_value"] = pd.to_numeric(match_df["home_squad_value"], errors="coerce")
                match_df["away_squad_value"] = pd.to_numeric(match_df["away_squad_value"], errors="coerce")

                match_df["home_squad_value_log"] = np.log1p(match_df["home_squad_value"])
                match_df["away_squad_value_log"] = np.log1p(match_df["away_squad_value"])

                match_df["squad_value_advantage"] = match_df["home_squad_value"] - match_df["away_squad_value"]
                match_df["squad_value_log_advantage"] = match_df["home_squad_value_log"] - match_df["away_squad_value_log"]

                for col in ["home_squad_value_log", "away_squad_value_log", "squad_value_log_advantage"]:
                    mean = match_df[col].mean(skipna=True)
                    std = match_df[col].std(skipna=True)
                    if std and not np.isnan(std):
                        match_df[col + "_z"] = (match_df[col] - mean) / std
                    else:
                        match_df[col + "_z"] = pd.NA

                match_df = match_df.drop(
                    columns=[
                        "home_squad_value",
                        "away_squad_value",
                        "home_squad_value_log",
                        "away_squad_value_log",
                        "squad_value_advantage",
                        "squad_value_log_advantage",
                    ],
                    errors="ignore",
                )
            
            match_df = match_df.drop(columns=["home_team_normalized", "away_team_normalized", "_match_date"], errors="ignore")
            match_df.to_csv(processed_file, index=False)
            
            print(f"Updated match data saved to {processed_file}")
            print("Added/updated columns: home_squad_value_log_z, away_squad_value_log_z, squad_value_log_advantage_z")
            
            print("\n" + "-"*70)
            print("COVERAGE ANALYSIS")
            print("-"*70)
            analyze_valuation_coverage(str(processed_file), after_date=FROM_DATE, valuation_csv=OUTPUT_CSV)
            
        except Exception as e:
            print(f"Error processing {processed_file.name}: {e}")
            continue

if __name__ == "__main__":
    main()
