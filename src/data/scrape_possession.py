import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

PROCESSED_BASE = Path("data/processed/processed_15_25_n10.csv")
RAW_OUT = Path("data/raw/fc_possession_raw.csv")

SEASON_URLS = {
    2015: "https://www.footballcritic.com/premier-league/season-2015-2016/matches/2/8978",
    2016: "https://www.footballcritic.com/premier-league/season-2016-2017/matches/2/11833",
    2017: "https://www.footballcritic.com/premier-league/season-2017-2018/matches/2/13294",
    2018: "https://www.footballcritic.com/premier-league/season-2018-2019/matches/2/16336",
    2019: "https://www.footballcritic.com/premier-league/season-2019-2020/matches/2/21558",
    2020: "https://www.footballcritic.com/premier-league/season-2020-2021/matches/2/41756",
    2021: "https://www.footballcritic.com/premier-league/season-2021-2022/matches/2/50885",
    2022: "https://www.footballcritic.com/premier-league/season-2022-2023/matches/2/65452",
    2023: "https://www.footballcritic.com/premier-league/season-2023-2024/matches/2/68731",
    2024: "https://www.footballcritic.com/premier-league/season-2024-2025/matches/2/72764",
    2025: "https://www.footballcritic.com/premier-league/season-2025-2026/matches/2/76035"
}

REQUEST_DELAY = 1.0

TEAM_NAME_MAP = {
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham Utd": "West Ham",
    "Hull City": "Hull",
    "Leeds Utd": "Leeds",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Nottingham Forest": "Nott'm Forest",
    "Notts Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "AFC Bournemouth": "Bournemouth",
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Leicester City": "Leicester",
}


def _mk_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            )
        }
    )
    return s


def _get_html(sess, url):
    r = sess.get(url, timeout=20)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(r.text, "lxml")


def _norm_team(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def _build_season_match_index(sess, season_url):
    soup = _get_html(sess, season_url)
    ul = soup.find("ul", class_=lambda c: c and "allMatches" in c.split())
    if not ul:
        print("  [WARN] no <ul class='allMatches'> for", season_url)
        return pd.DataFrame(columns=["date", "home_team", "away_team", "fc_url"])

    rows = []

    for li in ul.find_all("li"):
        link = li.find("a", href=lambda h: h and "/match-stats/" in h)
        if not link:
            continue

        fc_url = link["href"]
        if fc_url.startswith("/"):
            fc_url = "https://www.footballcritic.com" + fc_url

        time_tag = li.find("time", class_=lambda c: c and "match-date" in c.split())
        if not time_tag or not time_tag.has_attr("datetime"):
            continue
        date_iso = time_tag["datetime"][:10]

        team_boxes = li.find_all("div", class_=lambda c: c and "info-box" in c.split())
        if len(team_boxes) < 2:
            continue

        def _extract_team_text(box):
            span = box.find("span", class_="text hidden-xs")
            if span is not None:
                return span.get_text(strip=True)
            a = box.find("a")
            if a is not None:
                return a.get_text(strip=True)
            return None

        raw_home = _extract_team_text(team_boxes[0])
        raw_away = _extract_team_text(team_boxes[-1])
        if not raw_home or not raw_away:
            continue

        home_team = _norm_team(raw_home)
        away_team = _norm_team(raw_away)

        rows.append(
            {
                "date": date_iso,
                "home_team": home_team,
                "away_team": away_team,
                "fc_url": fc_url,
            }
        )

    if not rows:
        print("  [WARN] no matches parsed for", season_url)
        return pd.DataFrame(columns=["date", "home_team", "away_team", "fc_url"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home_team", "away_team"])
    print("  parsed", len(df), "matches from", season_url)
    return df


def _build_full_match_index(sess, season_urls):
    all_rows = []
    for year, url in sorted(season_urls.items()):
        print("Building index for season starting", year, "->", url)
        df = _build_season_match_index(sess, url)
        if df.empty:
            print("  (no fixtures found for", year, ")")
            continue
        df["season_start"] = year
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "fc_url"])
    out = pd.concat(all_rows, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    return out


def _extract_possession(soup):
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Ball\s+)?Possession", text, re.I)
    if not m:
        return None, None

    start = max(m.start() - 200, 0)
    end = min(m.end() + 200, len(text))
    window = text[start:end]

    percents = re.findall(r"(\d{1,3})\s*%", window)
    if len(percents) < 2:
        nums = re.findall(r"\b(\d{1,3})\b", window)
        if len(nums) >= 2:
            percents = nums[:2]

    if len(percents) < 2:
        return None, None

    try:
        h = float(percents[0])
        a = float(percents[1])
    except Exception:
        return None, None

    s = h + a
    if 90 <= s <= 110:
        h = h * 100.0 / s
        a = a * 100.0 / s

    return h, a


def _fetch_possession(sess, url):
    soup = _get_html(sess, url)
    return _extract_possession(soup)


def analyze_possession_coverage(processed_file, raw_df):
    try:
        df = pd.read_csv(processed_file)
    except Exception as e:
        print("[coverage] could not read", processed_file, ":", e)
        return

    print("\n" + "=" * 70)
    print("POSSESSION DATA COVERAGE ANALYSIS")
    print("=" * 70)
    print("Processed file:", processed_file)

    # Merge in raw possession columns
    tmp = df.merge(
        raw_df[
            [
                "date",
                "home_team",
                "away_team",
                "home_possession_pct",
                "away_possession_pct",
                "possession_diff",
            ]
        ],
        on=["date", "home_team", "away_team"],
        how="left",
        validate="m:1",
    )

    total = len(tmp)
    print("Total matches in this file:", total)

    has_home = tmp["home_possession_pct"].notna()
    has_away = tmp["away_possession_pct"].notna()
    has_both = has_home & has_away
    has_any = has_home | has_away

    n_any = int(has_any.sum())
    n_both = int(has_both.sum())

    print("\nFill rate (possession in raw scrape):")
    print(f"  any possession value: {n_any} / {total} ({100.0 * n_any / total if total else 0.0:.2f}%)")
    print(f"  both home & away: {n_both} / {total} ({100.0 * n_both / total if total else 0.0:.2f}%)")

    if n_both > 0:
        tmp2 = tmp.loc[has_both, ["home_possession_pct", "away_possession_pct"]].astype(float)
        sums = tmp2["home_possession_pct"] + tmp2["away_possession_pct"]
        tol = 1.0
        bad = sums[(sums - 100.0).abs() > tol]
        n_bad = int(bad.shape[0])
        print(f"\nSum-to-100 check (tolerance ±{tol}):")
        print(f"  rows with both sides but sum != 100±{tol}: {n_bad} ({100.0 * n_bad / n_both if n_both else 0.0:.2f}%)")

        if n_bad > 0:
            print("\n  Example rows with non-100 sums:")
            ex = tmp.loc[has_both].iloc[bad.index].head(5)
            for _, r in ex.iterrows():
                s_val = float(r["home_possession_pct"] + r["away_possession_pct"])
                print(f"   {r['date']} {r['home_team']} vs {r['away_team']}: {r['home_possession_pct']} + {r['away_possession_pct']} = {s_val:.2f}")

    print("=" * 70 + "\n")


def main():
    sess = _mk_session()

    print("Reading base processed match CSV...")
    base_df = pd.read_csv(PROCESSED_BASE)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.date.astype(str)

    print("Building FootballCritic match index...")
    index_df = _build_full_match_index(sess, SEASON_URLS)

    print("Merging index with base processed data...")
    merged = base_df.merge(
        index_df,
        on=["date", "home_team", "away_team"],
        how="left",
        validate="m:1",
    )

    missing = merged["fc_url"].isna().sum()
    if missing:
        print("Warning:", missing, "matches have no FootballCritic URL after merge")

    records = []
    for i, row in merged.iterrows():
        url = row.get("fc_url")
        if not isinstance(url, str) or not url:
            continue

        print(f"[{i + 1}/{len(merged)}] {row['date']} {row['home_team']} vs {row['away_team']} -> {url}")
        try:
            h, a = _fetch_possession(sess, url)
        except Exception as e:
            print(f"  error fetching possession: {e}")
            h, a = None, None

        if h is not None and a is not None:
            diff = h - a
        else:
            diff = None

        rec = {
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "fc_url": url,
            "home_possession_pct": h,
            "away_possession_pct": a,
            "possession_diff": diff
        }
        records.append(rec)

    if not records:
        print("No possession records scraped – nothing to write.")
        return

    raw_df = pd.DataFrame(records)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Writing raw possession data to", RAW_OUT)
    raw_df.to_csv(RAW_OUT, index=False)

    # Now merge possession_diff into ALL processed CSVs in data/processed
    processed_dir = Path("data/processed")
    csv_files = sorted(processed_dir.glob("*.csv"))

    for processed_file in csv_files:
        print("\n" + "=" * 70)
        print("Processing:", processed_file.name)
        print("=" * 70)

        try:
            df = pd.read_csv(processed_file)
            print("Loaded match data:", len(df), "rows")

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

            if "possession_diff" in df.columns:
                df = df.drop(columns=["possession_diff"])

            df = df.merge(
                raw_df[["date", "home_team", "away_team", "possession_diff"]],
                on=["date", "home_team", "away_team"],
                how="left",
                validate="m:1",
            )

            df.to_csv(processed_file, index=False)
            print("Updated match data saved to", processed_file)
            print("Added/updated column: possession_diff")

            # Coverage analysis
            print("\n" + "-" * 70)
            print("COVERAGE ANALYSIS (using raw possession data)")
            print("-" * 70)
            analyze_possession_coverage(str(processed_file), raw_df)

        except Exception as e:
            print("Error processing", processed_file.name, ":", e)
            continue


if __name__ == "__main__":
    main()
