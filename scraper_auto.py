import asyncio
import pandas as pd
from google_play_scraper import reviews, Sort
import time
import os

# =========================
# APLIKASI OJOL
# =========================
APPS = {
    "Gojek": "com.gojek.app",
    "Grab": "com.grabtaxi.passenger",
    "Maxim": "com.taxsee.taxsee",
    "inDrive": "sinet.startup.inDriver",
    "Anterin": "com.anterin.dn",
    "Nujek": "com.nujek.user",
}

# =========================
# FILTER TAHUN
# =========================
START_YEAR = 2020
END_YEAR = 2026


# =========================
# SCRAPER FUNCTION
# =========================
async def scrape_app(app_name, app_id, max_data=10000):
    print(f"🚀 Scraping {app_name}")

    data = []
    token = None
    loop = asyncio.get_event_loop()

    while True:
        result, token = await loop.run_in_executor(
            None,
            lambda: reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                continuation_token=token
            )
        )

        if not result:
            break

        for r in result:
            try:
                # =========================
                # FIX TIMEZONE + PARSING AMAN
                # =========================
                date = pd.to_datetime(r['at'], errors='coerce', utc=True)

                if pd.isna(date):
                    continue

                date = date.tz_convert(None)
                year = int(date.year)

                # =========================
                # FILTER VALID YEAR ONLY
                # =========================
                if START_YEAR <= year <= END_YEAR:
                    data.append({
                        'app': app_name,
                        'date': date,
                        'year': year,
                        'content': r['content'],
                        'score': r['score']
                    })

            except:
                continue

        if not token or len(data) >= max_data:
            break

    return pd.DataFrame(data)


# =========================
# MAIN RUN
# =========================
async def main():
    start = time.time()

    tasks = [
        scrape_app(name, aid)
        for name, aid in APPS.items()
    ]

    results = await asyncio.gather(*tasks)

    final_df = pd.concat(results, ignore_index=True)

    # =========================
    # FINAL CLEAN FILTER (DOUBLE SAFETY)
    # =========================
    final_df = final_df[
        (final_df['year'] >= START_YEAR) &
        (final_df['year'] <= END_YEAR)
    ]

    # =========================
    # SAVE DATASET
    # =========================
    os.makedirs("data", exist_ok=True)
    final_df.to_csv("data/ojol_dataset.csv", index=False)

    print("===================================")
    print(f"✅ TOTAL DATA: {len(final_df)}")
    print(final_df['year'].value_counts().sort_index())
    print(f"⚡ DONE in {time.time() - start:.2f}s")
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main())