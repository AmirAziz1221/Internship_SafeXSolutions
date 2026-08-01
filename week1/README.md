# Rule-Based Lead Scoring Model

## Objective
Score and rank 25 mock leads by likelihood to convert, using a transparent,
rule-based system (no ML) so the logic is easy to explain and adjust.

## Files
- `lead_scoring.py` — full script: mock dataset, scoring logic, ranking, CSV export
- `ranked_leads.csv` — output: all 25 leads ranked highest to lowest score

## Scoring Criteria (100 points total)

| Criterion | Max Points | Why it matters |
|---|---|---|
| **Industry Fit** | 20 | Software/Finance/Healthcare are the target verticals for this mock product; scored higher than Retail/Education/Manufacturing. |
| **Company Size** | 20 | Bigger companies → bigger budgets → higher deal value. Tiered by employee count (250+, 100+, 50+, 20+, <20). |
| **Engagement Level** | 20 | Combines email opens (out of 5) and website visits (capped at 10). Direct proxy for interest. |
| **Job Title / Authority** | 20 | Senior titles (CTO, CFO, Founder, VP, Director) score highest since they can approve purchases; mid-level (Manager, Analyst) score medium; junior roles score low. |
| **Lead Source Quality** | 10 | Referrals convert best historically, then Webinar, Website Form, Trade Show, Cold Outreach last. |
| **Recency of Activity** | 10 | Leads active in the last 3 days score highest; activity older than 30 days scores 0 — interest fades fast. |

## How the score is derived
Each lead is scored independently on all 6 criteria using simple lookup
tables and thresholds (e.g. `company_size >= 250 → 20 pts`). The 6 scores
are summed into a `total_score` out of 100. No weighting is hidden — every
rule lives in one small function per criterion in `lead_scoring.py`, so
anyone can open the file and see exactly why a lead scored the way it did.

## Grading bands
- **Hot** (75–100): sales-ready, prioritize immediate outreach
- **Warm** (50–74): nurture with content/follow-up, revisit in 1–2 weeks
- **Cold** (0–49): low priority, keep in general marketing list

## Result summary (this run)
- 10 Hot leads, 7 Warm, 8 Cold
- Top lead: Ayesha Khan (Software, CTO, Referral, 94/100) — decision-maker
  at a strong-fit company who engaged very recently through a high-trust
  channel.
- Bottom lead: Danish Karim (Education, Intern, Cold Outreach, 17/100) —
  wrong industry, no authority, no engagement, and inactive for 60 days.

## How to run
```bash
python3 lead_scoring.py
```
Prints the ranked table to console and writes `ranked_leads.csv`.

## Possible extensions (not built, just noted for future work)
- Replace fixed point tables with configurable weights in a JSON/YAML config
- Add a decay function for recency instead of step buckets
- Feed the rule-based scores into a logistic regression model later to
  compare "rules vs. learned weights" — natural next step once real
  conversion outcome data exists.
