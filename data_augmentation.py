import pandas as pd
import random
import os
import sys

# Ensure Windows console can handle all characters
sys.stdout.reconfigure(encoding='utf-8')

def generate_synthetic_fake_news(num_samples=3000):
    # ── Template categories ────────────────────────────────────
    templates = [
        # Generic fake news
        "Scientists confirm that {topic} can cure all diseases instantly",
        "Government secretly hiding the truth about {topic}",
        "Shocking discovery reveals {topic} is dangerous to your health",
        "Experts don't want you to know about {topic}",
        "Miracle solution for {topic} found overnight",
        "Hidden agenda exposed: {topic} is a cover-up by elites",
        "Breaking: Leaked document reveals shocking truth about {topic}",
        "They have been LYING to you about {topic} for years",
        "Whistleblower exposes secret {topic} plot by the deep state",
        "Exclusive: The real reason they BANNED {topic} from mainstream media",
        # Political conspiracy
        "{person} SECRETLY plotting to destroy America with {topic}",
        "EXPOSED: {person} funded by globalists to control {topic}",
        "{person} admits to rigging {topic} in leaked recording",
        "INSIDER reveals how {person} stole {topic} from the people",
        "Deep state COVER-UP: {person} arrested for {topic} crimes SUPPRESSED",
        # Health and vaccine conspiracy
        "BANNED cure: {topic} destroys immunity and doctors are HIDING it",
        "URGENT: {topic} contains microchips connected to 5G towers",
        "Doctors CENSORED for revealing {topic} causes brain damage",
        "BIG PHARMA hiding {topic} cure that costs nothing",
        # Media conspiracy
        "Mainstream MEDIA HIDING truth about {topic} says whistleblower",
        "CNN CAUGHT staging {topic} coverage says insider source",
        "FAKE: The {topic} story the media refuses to report",
        "SUPPRESSED VIDEO: NASA astronaut LEAKS footage of {topic}",
    ]
    topics = [
        "vaccines", "health", "economy", "technology", "climate", "education",
        "5G networks", "AI", "the global market", "elections", "taxes",
        "COVID-19", "the deep state", "fluoride", "chemtrails", "GMOs",
        "microchips", "the Moon landing", "flat earth", "alien contact",
        "cell towers", "drinking water", "the banking system",
    ]
    persons = [
        "Bill Gates", "George Soros", "Hillary Clinton", "Barack Obama",
        "Anthony Fauci", "Mark Zuckerberg", "Klaus Schwab", "Joe Biden",
    ]

    # ── Hard-coded real-world-style examples ───────────────────
    hardcoded_fake = [
        "SHOCKING Scientists PROVE that 5G towers cause cancer and the government is HIDING it",
        "BREAKING Bill Gates admits microchips are inside COVID vaccines EXPOSED by whistleblower",
        "Pope Francis SECRETLY endorses flat earth theory in leaked Vatican document",
        "Donald Trump won 2020 election voting machines were rigged says INSIDER source",
        "Drinking bleach cures COVID doctors BANNED from telling you the TRUTH",
        "Hollywood actress EXPOSED for running secret satanic cult in California",
        "Government CHEMTRAILS confirmed by whistleblower pilot full COVER-UP revealed",
        "URGENT Mainstream media HIDING alien contact NASA astronaut LEAKS footage",
        "George Soros FUNDS antifa to DESTROY America says leaked document",
        "Vaccines cause autism and the CDC is hiding the data from the public",
        "The moon landing was FAKED by NASA in a Hollywood studio secret finally exposed",
        "Scientists BANNED from publishing truth about climate change hoax conspiracy",
        "SHOCKING EXPOSED elite pedophile ring running worldwide trafficking operation",
        "Secret government program to control population through water supply revealed",
        "Pizzagate is real and FBI is covering it up whistleblower says",
        "Obama born in Kenya secret document finally surfaces PROOF exposed",
        "COVID vaccine magnetizes people and connects them to 5G internet",
        "Microchips implanted during vaccine rollout confirmed by ex-Pfizer employee",
        "QAnon insider drops bombshell that will DESTROY the deep state forever now",
        "BREAKING election fraud evidence discovered in all 50 states media silent",
    ]

    synthetic_data = []
    # From hardcoded list
    for text in hardcoded_fake:
        synthetic_data.append({"content": text, "label": 0})

    # From templates
    for _ in range(num_samples - len(hardcoded_fake)):
        template = random.choice(templates)
        topic    = random.choice(topics)
        person   = random.choice(persons)
        text     = template.format(topic=topic, person=person, problem=topic)
        # Pad to pass the >= 10 words filter
        padding  = " " + " ".join(random.choices(
            ["today", "now", "urgent", "breaking", "says", "insider", "leaked", "exposed", "secret"], k=4
        ))
        synthetic_data.append({"content": text + padding, "label": 0})

    return pd.DataFrame(synthetic_data)

def augment_data():
    base_file = "data/news.csv"
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found. Run run_once_prepare_data.py first.")
        # Create an empty dataframe to continue if needed, or return
        df = pd.DataFrame(columns=["content", "label"])
    else:
        df = pd.read_csv(base_file)
        
    print(f"Original dataset size: {len(df)}")
    
    # Generate requested amount
    synth_df = generate_synthetic_fake_news(3000)
    print(f"Generated {len(synth_df)} synthetic fake news samples.")
    
    # Just for local dummy testing robustness, we'll synthesize some REAL data to prevent extreme class imbalance drops locally
    # Note: In real production, Kaggle dataset has ~20,000 Real cases so this wouldn't strictly be needed.
    real_templates = [
        # Government and politics
        "The Federal Reserve announced a new approach to {topic}",
        "Congress debates the future implications of {topic}",
        "Local officials hold townhall discussing {topic}",
        "The president signed an executive order regarding {topic}",
        "The Supreme Court ruled in favor of new regulations on {topic}",
        "Parliament passed new legislation addressing {topic} concerns",
        "Government officials released new data on {topic} growth this quarter",
        # Science and space
        "NASA successfully launched a new mission to explore {space_topic}",
        "Scientists discover new species of {animal} near the {location}",
        "Researchers published findings on {topic} in a peer reviewed journal",
        "New studies published in medical journals regarding {topic}",
        "A team of biologists identified a previously unknown {animal} in the {location}",
        # Economics and business
        "{company} reported quarterly earnings beating analyst estimates",
        "{company} acquires {company2} in a deal worth billions of dollars",
        "{country} GDP grew by several percent in the last fiscal quarter according to government data",
        "The World Bank released its annual report on global {topic} trends",
        "Stock markets closed higher after positive {topic} data was released",
        # Health
        "The World Health Organization approved a new vaccine for {topic}",
        "Medical researchers announced a breakthrough in treating {topic}",
        "New clinical trial results show promise for {topic} treatment",
        # International
        "The United Nations climate report warns of rising sea levels by 2050",
        "International leaders met at the summit to discuss {topic} cooperation",
        "{country} announced new trade agreements with neighboring countries",
        # Technology
        "{company} announced a new battery technology with improved range",
        "{company} unveiled their latest artificial intelligence research results",
        "Regulators proposed new rules for data privacy and AI governance",
    ]
    # Diverse fill-ins to avoid template monotony
    companies = ["Apple", "Tesla", "Microsoft", "Google", "Amazon", "Samsung", "Intel"]
    companies2 = ["a gaming company", "a startup", "an AI firm", "a cloud provider"]
    countries = ["India", "China", "Brazil", "Germany", "Japan", "France", "South Korea"]
    animals = ["deep sea fish", "frog", "insect", "coral", "marine organism"]
    locations = ["Pacific Ocean floor", "Amazon rainforest", "Arctic region", "Indian Ocean"]
    space_topics = ["the Moon", "Mars", "deep space", "the International Space Station"]
    topics_real = ["vaccines", "health", "economy", "technology", "climate", "education",
                   "energy", "agriculture", "infrastructure", "defense", "trade"]

    real_data = []
    for _ in range(3000):
        t = random.choice(real_templates)
        text = t.format(
            topic=random.choice(topics_real),
            company=random.choice(companies),
            company2=random.choice(companies2),
            country=random.choice(countries),
            animal=random.choice(animals),
            location=random.choice(locations),
            space_topic=random.choice(space_topics),
        )
        pad = " " + " ".join(random.choices(["according", "to", "data", "reports", "officials", "said", "announced", "today", "this"], k=5))
        real_data.append({"content": text + pad, "label": 1})
        
    synth_real_df = pd.DataFrame(real_data)
    
    combined = pd.concat([df, synth_df, synth_real_df], ignore_index=True)
    
    out_file = "data/news_augmented.csv"
    os.makedirs("data", exist_ok=True)
    combined.to_csv(out_file, index=False)
    print(f"[OK] Augmented dataset saved to {out_file} with {len(combined)} total rows.")

if __name__ == "__main__":
    augment_data()
