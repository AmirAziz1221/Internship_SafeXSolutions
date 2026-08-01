"""
Rule-Based Lead Scoring Model
=============================
Scores and ranks mock leads by likelihood to convert, using 6 weighted
criteria. Pure rule-based logic (no ML) -- transparent and easy to explain
to a non-technical stakeholder.

Author: Amir Aziz
"""

import pandas as pd

# -----------------------------------------------------------------------
# STEP 1: MOCK LEAD DATASET
# -----------------------------------------------------------------------
# In a real scenario this would come from a CRM export (HubSpot, Salesforce,
# Zoho, etc.). Here we hand-build 25 realistic leads covering a spread of
# industries, company sizes, engagement levels, job titles, lead sources,
# and recency -- so the scoring logic has enough variance to be meaningful.

leads = [
    # name, industry, company_size, emails_opened(/5), website_visits, job_title, lead_source, days_since_last_activity
    ("Ayesha Khan",      "Software",     120,  5,  8, "CTO",              "Referral",        1),
    ("Bilal Ahmed",      "Retail",       15,   1,  1, "Sales Associate",  "Cold Outreach",   40),
    ("Sana Malik",       "Software",     45,   4,  6, "Product Manager",  "Webinar",         3),
    ("Usman Tariq",      "Manufacturing",300,  2,  3, "Operations Head",  "Trade Show",      12),
    ("Fatima Noor",      "Healthcare",   80,   5,  9, "IT Director",      "Referral",        2),
    ("Hamza Sheikh",     "Finance",      500,  3,  4, "VP Engineering",   "Website Form",    5),
    ("Zainab Riaz",      "Software",     8,    2,  2, "Founder",          "Cold Outreach",   20),
    ("Ali Raza",         "Education",    60,   1,  0, "Teacher",          "Website Form",    35),
    ("Mariam Farooq",    "Software",     220,  5,  7, "Head of Data",     "Webinar",         1),
    ("Omar Sultan",      "Retail",       35,   3,  5, "Store Manager",    "Referral",        7),
    ("Hira Younis",      "Manufacturing",150,  4,  6, "Plant Manager",    "Trade Show",       4),
    ("Kashif Iqbal",     "Software",     90,   2,  2, "Software Engineer","Cold Outreach",   18),
    ("Nimra Saeed",      "Finance",      40,   5, 10, "CFO",              "Referral",         1),
    ("Faisal Chaudhry",  "Healthcare",   25,   1,  1, "Nurse",            "Website Form",    50),
    ("Rabia Aslam",      "Software",     600,  4,  5, "VP Product",       "Webinar",          6),
    ("Danish Karim",     "Education",    10,   0,  0, "Intern",           "Cold Outreach",   60),
    ("Sara Baig",        "Software",     130,  5,  8, "CTO",              "Referral",         2),
    ("Waleed Anjum",     "Manufacturing",70,   2,  3, "Quality Analyst",  "Website Form",    22),
    ("Iqra Shahid",      "Finance",      210,  3,  4, "Finance Manager",  "Trade Show",       9),
    ("Junaid Latif",     "Retail",       5,    1,  0, "Owner",            "Cold Outreach",   45),
    ("Alina Zafar",      "Software",     55,   5,  9, "Product Owner",    "Webinar",          1),
    ("Tariq Mehmood",    "Healthcare",   400,  4,  6, "IT Director",      "Referral",         3),
    ("Nadia Yousaf",     "Software",     18,   3,  4, "Founder",          "Website Form",    14),
    ("Asad Bhatti",      "Manufacturing",25,   1,  1, "Technician",       "Cold Outreach",   55),
    ("Mehak Kamal",      "Finance",      95,   5,  7, "VP Finance",       "Referral",         2),
]

df = pd.DataFrame(leads, columns=[
    "name", "industry", "company_size", "emails_opened",
    "website_visits", "job_title", "lead_source", "days_since_last_activity"
])

# -----------------------------------------------------------------------
# STEP 2: SCORING CRITERIA (6 criteria, 100 points total)
# -----------------------------------------------------------------------
# Each criterion is scored independently on its own point scale, then
# summed. Weights reflect what actually predicts conversion for a
# B2B SaaS-style pipeline: engagement and authority matter most.
#
#   1. Industry Fit          -> 20 pts
#   2. Company Size          -> 20 pts
#   3. Engagement Level       -> 20 pts   (email opens + site visits)
#   4. Job Title / Authority  -> 20 pts   (decision-making power)
#   5. Lead Source Quality    -> 10 pts
#   6. Recency of Activity    -> 10 pts

TARGET_INDUSTRIES = {"Software": 20, "Finance": 16, "Healthcare": 14,
                      "Manufacturing": 10, "Retail": 8, "Education": 6}

SOURCE_SCORES = {"Referral": 10, "Webinar": 8, "Website Form": 6, "Trade Show": 5, "Cold Outreach": 2}

SENIOR_TITLES = ["CTO", "CFO", "Founder", "VP", "Head of", "Director"]
MID_TITLES = ["Manager", "Product Owner", "Product Manager", "Analyst"]


def score_industry(industry):
    return TARGET_INDUSTRIES.get(industry, 5)


def score_company_size(size):
    if size >= 250:
        return 20
    elif size >= 100:
        return 16
    elif size >= 50:
        return 12
    elif size >= 20:
        return 8
    else:
        return 4


def score_engagement(emails_opened, website_visits):
    # emails_opened is out of 5, website_visits roughly out of 10
    email_score = min(emails_opened, 5) * 2          # up to 10
    visit_score = min(website_visits, 10)             # up to 10
    return email_score + visit_score                  # up to 20


def score_job_title(title):
    if any(t.lower() in title.lower() for t in SENIOR_TITLES):
        return 20
    elif any(t.lower() in title.lower() for t in MID_TITLES):
        return 12
    else:
        return 5


def score_lead_source(source):
    return SOURCE_SCORES.get(source, 3)


def score_recency(days):
    if days <= 3:
        return 10
    elif days <= 7:
        return 8
    elif days <= 14:
        return 5
    elif days <= 30:
        return 2
    else:
        return 0


# -----------------------------------------------------------------------
# STEP 3: APPLY SCORING LOGIC
# -----------------------------------------------------------------------
df["industry_score"]   = df["industry"].apply(score_industry)
df["size_score"]       = df["company_size"].apply(score_company_size)
df["engagement_score"] = df.apply(lambda r: score_engagement(r["emails_opened"], r["website_visits"]), axis=1)
df["title_score"]      = df["job_title"].apply(score_job_title)
df["source_score"]     = df["lead_source"].apply(score_lead_source)
df["recency_score"]    = df["days_since_last_activity"].apply(score_recency)

df["total_score"] = (
    df["industry_score"] + df["size_score"] + df["engagement_score"]
    + df["title_score"] + df["source_score"] + df["recency_score"]
)


def grade(score):
    if score >= 75:
        return "Hot"
    elif score >= 50:
        return "Warm"
    else:
        return "Cold"


df["lead_grade"] = df["total_score"].apply(grade)

# -----------------------------------------------------------------------
# STEP 4: RANK AND EXPORT
# -----------------------------------------------------------------------
ranked = df.sort_values("total_score", ascending=False).reset_index(drop=True)
ranked.insert(0, "rank", ranked.index + 1)

output_cols = [
    "rank", "name", "industry", "company_size", "job_title", "lead_source",
    "days_since_last_activity", "industry_score", "size_score",
    "engagement_score", "title_score", "source_score", "recency_score",
    "total_score", "lead_grade"
]
ranked = ranked[output_cols]

ranked.to_csv("ranked_leads.csv", index=False)

print(ranked.to_string(index=False))
print(f"\nSaved {len(ranked)} ranked leads to ranked_leads.csv")
print(f"\nGrade distribution:\n{ranked['lead_grade'].value_counts()}")
