#!/usr/bin/env python3
"""Seed script — registers two demo clients and ingests sample documents.

Usage:
    docker compose exec app python scripts/seed_data.py
    # or locally:
    python scripts/seed_data.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.clients.manager import register
from app.ingestion.intake import ingest_document

# ── Demo clients ──────────────────────────────────────────────────────────────
CLIENTS = [
    {"client_id": "acme_corp", "display_name": "ACME Corporation", "notes": "Enterprise software client"},
    {"client_id": "techstart", "display_name": "TechStart Inc", "notes": "Early-stage SaaS startup"},
]

# ── Sample documents per client ───────────────────────────────────────────────
DOCS = {
    "acme_corp": [
        {
            "source": "employee_handbook.txt",
            "text": """
ACME Corporation Employee Handbook

Welcome to ACME Corporation. This handbook outlines our policies, procedures, and culture.

Work Hours and Remote Policy
Standard work hours are 9 AM to 6 PM in your local timezone. We support flexible working arrangements.
Employees may work remotely up to 3 days per week with manager approval. Core collaboration hours
are 10 AM to 3 PM when all team members should be reachable.

Leave Policy
Annual leave entitlement is 25 days per year. Employees must give at least 2 weeks notice for
leave requests longer than 5 days. Unused leave can be carried over up to 10 days into the next year.
Sick leave is unlimited but requires a doctor's note after 3 consecutive days.

Expense Reimbursement
All business expenses must be submitted within 30 days of being incurred. Receipts are required
for any expense over $25. Travel expenses require pre-approval from your department head.
Software subscriptions must go through the IT procurement process.

Performance Reviews
Performance reviews are conducted bi-annually in June and December. Employees are rated on
a 1-5 scale across five competencies: technical excellence, collaboration, communication,
initiative, and impact. Ratings of 4 or 5 qualify for merit increases and bonus eligibility.

Code of Conduct
ACME maintains a zero-tolerance policy for harassment, discrimination, and dishonesty.
Violations should be reported to HR or through the anonymous ethics hotline at ethics@acme.com.
All reports are treated confidentially and investigated within 10 business days.
""",
        },
        {
            "source": "technical_standards.txt",
            "text": """
ACME Engineering Standards v3.1

Code Review Process
All code changes require at least one approved review before merging to main. Reviews should
be completed within 24 hours of submission. Reviewers should check for correctness, test
coverage (minimum 80%), security vulnerabilities, and adherence to style guides.

CI/CD Pipeline
Our pipeline runs linting, unit tests, integration tests, and security scans on every pull request.
Deployments to staging are automatic on merge to main. Production deployments require a manual
approval gate and must be scheduled during the deployment window (Tuesdays and Thursdays, 2-4 PM).

Incident Response
Severity 1 incidents require a response within 15 minutes and all-hands until resolved.
Severity 2 incidents require acknowledgment within 1 hour. All incidents require a post-mortem
within 5 business days. Post-mortems follow a blameless format and focus on systemic improvements.

Data Handling
Production data must never be copied to local machines or development environments.
All data at rest must be encrypted using AES-256. Data in transit must use TLS 1.3 or higher.
Personal data (PII) must be identified, tagged, and handled according to our GDPR compliance policy.

API Standards
All APIs must be versioned (e.g., /api/v1/). Deprecation requires 6 months notice.
Rate limiting is enforced at 1000 requests/minute per API key. All endpoints must return
standard error responses with error codes and human-readable messages.
""",
        },
    ],
    "techstart": [
        {
            "source": "product_docs.txt",
            "text": """
TechStart Platform Documentation

Getting Started
TechStart is a B2B SaaS platform for automated financial reporting. Sign up at app.techstart.io
and complete the onboarding wizard to connect your accounting software. We support QuickBooks,
Xero, FreshBooks, and direct CSV imports.

Integrations
Connect TechStart to your existing tools using our OAuth2 integrations. Available integrations
include Slack (for report delivery), Google Sheets (for data export), Salesforce (for revenue data),
and Stripe (for real-time revenue metrics). New integrations are added monthly based on user votes.

Reports
TechStart generates seven standard report types: P&L statement, balance sheet, cash flow forecast,
burn rate analysis, unit economics dashboard, cohort analysis, and investor update pack.
Custom reports can be built using our drag-and-drop report builder. All reports update in real-time
as new transactions arrive.

Pricing
TechStart is priced based on monthly active users and connected data sources. The Starter plan
covers up to 5 users and 3 integrations at $99/month. The Growth plan covers up to 25 users and
unlimited integrations at $299/month. Enterprise pricing is custom and includes dedicated support.

Security and Compliance
TechStart is SOC 2 Type II certified. We use bank-level encryption for all financial data.
Two-factor authentication is required for all accounts. Data is stored in EU and US regions
based on customer preference. GDPR and CCPA compliance documentation is available on request.
""",
        },
        {
            "source": "faq.txt",
            "text": """
TechStart Frequently Asked Questions

How do I reset my password?
Go to app.techstart.io/forgot-password and enter your email address. You will receive a reset
link within 5 minutes. The link expires after 24 hours. If you do not receive the email,
check your spam folder or contact support@techstart.io.

Can I export my data?
Yes. All reports can be exported as PDF, Excel, or CSV. Raw transaction data can be exported
as CSV from the Data tab. For bulk exports or API access to your data, contact our support team.
Data exports are available on Growth and Enterprise plans.

What happens if I exceed my plan limits?
If you exceed your monthly active user limit, additional users will be billed at $15/user/month.
If you need to connect more data sources than your plan allows, you will be prompted to upgrade.
We send email warnings at 80% and 95% of your limits.

How does the cash flow forecast work?
TechStart's cash flow forecast uses your historical transaction data combined with your confirmed
future obligations (subscriptions, contracts, payroll) to project cash position up to 12 months ahead.
The model updates daily and shows confidence intervals based on historical variance.

Is there a free trial?
Yes. All new accounts start with a 14-day free trial of the Growth plan with no credit card required.
At the end of the trial, you can choose any plan or cancel with no charge. Trial data is preserved
if you choose to subscribe.
""",
        },
    ],
}


def main():
    print("Seeding demo clients and documents...\n")

    for client_data in CLIENTS:
        cid = client_data["client_id"]
        register(**client_data)
        print(f"✓ Registered client: {cid}")

        docs = DOCS.get(cid, [])
        for doc in docs:
            result = ingest_document(
                text=doc["text"],
                client_id=cid,
                source=doc["source"],
            )
            print(
                f"  ↳ Ingested '{doc['source']}': "
                f"{result.stored_chunks} chunks stored, "
                f"{result.duplicate_chunks} dupes, "
                f"{result.anomalous_chunks} anomalous"
            )

    print("\nSeed complete. You can now query:")
    for c in CLIENTS:
        print(f"  • {c['client_id']} ({c['display_name']})")


if __name__ == "__main__":
    main()
