"""
Phase 0 — Evaluation Benchmark for AI Analytics Consultant.

This is the fixed measurement instrument for the entire project.
Defined and verified BEFORE any agent code exists, per the design doc's
build-order principle (Section 7: build the benchmark before the agents).

Every question has a verified correct answer, computed directly against
analytics.db and confirmed by hand. Do not modify ground_truth values
without re-verifying against the database — these are the answer key.

Tiers:
  1 — Direct lookup       (single table, no joins)
  2 — Aggregation         (group by, single join)
  3 — Multi-table joins   (3+ tables, correct join logic required)
  4 — Ambiguous / business logic (no single correct SQL, judged on reasoning)
  5 — Visualization + modeling (full pipeline, not yet scoreable until
      Chart Agent and Model Agent exist — placeholders for now)
"""

BENCHMARK = [
    # ── Tier 1: Direct lookup ────────────────────────────────────────────
    {
        "id": "T1-01",
        "tier": 1,
        "question": "How many customers do we have?",
        "ground_truth": 60,
        "ground_truth_sql": "SELECT COUNT(*) FROM customers",
        "answer_type": "exact_number",
    },
    {
        "id": "T1-02",
        "tier": 1,
        "question": "How many products do we sell?",
        "ground_truth": 18,
        "ground_truth_sql": "SELECT COUNT(*) FROM products",
        "answer_type": "exact_number",
    },
    {
        "id": "T1-03",
        "tier": 1,
        "question": "How many support tickets are currently open?",
        "ground_truth": 17,
        "ground_truth_sql": "SELECT COUNT(*) FROM support_tickets WHERE status = 'Open'",
        "answer_type": "exact_number",
    },
    {
        "id": "T1-04",
        "tier": 1,
        "question": "How many critical-priority support tickets have we had?",
        "ground_truth": 12,
        "ground_truth_sql": "SELECT COUNT(*) FROM support_tickets WHERE priority = 'Critical'",
        "answer_type": "exact_number",
    },

    # ── Tier 2: Aggregation ──────────────────────────────────────────────
    {
        "id": "T2-01",
        "tier": 2,
        "question": "What is our total revenue from completed orders?",
        "ground_truth": 4859003.00,
        "ground_truth_sql": """SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2)
            FROM order_items oi JOIN orders o ON oi.order_id = o.order_id
            WHERE o.status = 'Completed'""",
        "answer_type": "exact_number",
        "tolerance": 1.00,  # allow $1 rounding variance
    },
    {
        "id": "T2-02",
        "tier": 2,
        "question": "What is our revenue broken down by customer segment?",
        "ground_truth": {"SMB": 2167960.00, "Enterprise": 1516995.00, "Startup": 1174048.00},
        "ground_truth_sql": """SELECT c.segment, ROUND(SUM(oi.quantity*oi.unit_price),2)
            FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
            JOIN customers c ON o.customer_id=c.customer_id
            WHERE o.status='Completed' GROUP BY c.segment ORDER BY 2 DESC""",
        "answer_type": "grouped_values",
        "tolerance": 1.00,
    },
    {
        "id": "T2-03",
        "tier": 2,
        "question": "Which product category generates the most revenue?",
        "ground_truth": "Services",
        "ground_truth_sql": """SELECT p.category, ROUND(SUM(oi.quantity*oi.unit_price),2) as rev
            FROM order_items oi JOIN products p ON oi.product_id=p.product_id
            JOIN orders o ON oi.order_id=o.order_id WHERE o.status='Completed'
            GROUP BY p.category ORDER BY rev DESC LIMIT 1""",
        "answer_type": "single_value",
    },
    {
        "id": "T2-04",
        "tier": 2,
        "question": "What is the average order value for completed orders?",
        "ground_truth": 21595.57,
        "ground_truth_sql": """SELECT ROUND(AVG(order_total),2) FROM (
            SELECT o.order_id, SUM(oi.quantity*oi.unit_price) as order_total
            FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
            WHERE o.status='Completed' GROUP BY o.order_id)""",
        "answer_type": "exact_number",
        "tolerance": 1.00,
    },

    # ── Tier 3: Multi-table joins ────────────────────────────────────────
    {
        "id": "T3-01",
        "tier": 3,
        "question": "Who are our top 5 customers by total revenue?",
        "ground_truth": [
            "Gekko & Co Inc 43", "Initech Holdings 7", "Umbrella Co 2",
            "Vandelay Partners 49", "Black Mesa Partners 5",
        ],
        "ground_truth_sql": """SELECT c.company_name, ROUND(SUM(oi.quantity*oi.unit_price),2) as rev
            FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
            JOIN customers c ON o.customer_id=c.customer_id WHERE o.status='Completed'
            GROUP BY c.company_name ORDER BY rev DESC LIMIT 5""",
        "answer_type": "ranked_list",
    },
    {
        "id": "T3-02",
        "tier": 3,
        "question": "How many customers have never opened a support ticket?",
        "ground_truth": 10,
        "ground_truth_sql": """SELECT COUNT(*) FROM customers c
            WHERE c.customer_id NOT IN (SELECT DISTINCT customer_id FROM support_tickets)""",
        "answer_type": "exact_number",
        "note": "Tests correct use of NOT IN / anti-join logic across two tables.",
    },
    {
        "id": "T3-03",
        "tier": 3,
        "question": "Which region has had the most critical-priority support tickets?",
        "ground_truth": "West",
        "ground_truth_sql": """SELECT c.region, COUNT(*) as cnt
            FROM support_tickets t JOIN customers c ON t.customer_id=c.customer_id
            WHERE t.priority='Critical' GROUP BY c.region ORDER BY cnt DESC LIMIT 1""",
        "answer_type": "single_value",
    },

    # ── Tier 4: Ambiguous / business logic ──────────────────────────────
    {
        "id": "T4-01",
        "tier": 4,
        "question": "Who are our best customers?",
        "ground_truth": None,
        "answer_type": "reasoning",
        "scoring_note": (
            "No single correct SQL. Score on whether the agent (a) states an "
            "explicit definition of 'best' — e.g. total revenue, order frequency, "
            "low support burden — and (b) the SQL it writes actually matches "
            "the definition it stated."
        ),
    },
    {
        "id": "T4-02",
        "tier": 4,
        "question": "Which customers are at risk of churning?",
        "ground_truth": None,
        "answer_type": "reasoning",
        "scoring_note": (
            "No ground truth — no explicit churn label in this dataset. Score on "
            "whether the agent identifies a reasonable proxy (e.g. no recent orders "
            "combined with open critical tickets) and explains its reasoning rather "
            "than fabricating a definitive answer."
        ),
    },
    {
        "id": "T4-03",
        "tier": 4,
        "question": "Is our support team keeping up with demand?",
        "ground_truth": None,
        "answer_type": "reasoning",
        "scoring_note": (
            "Score on whether the agent looks at relevant signals — proportion of "
            "open vs. closed tickets, time-to-resolution, ticket volume trend — "
            "rather than guessing without querying the data."
        ),
    },

    # ── Tier 5: Visualization + modeling (placeholders) ─────────────────
    {
        "id": "T5-01",
        "tier": 5,
        "question": "Show a chart of monthly revenue over time.",
        "ground_truth": None,
        "answer_type": "chart",
        "scoring_note": "Not scoreable until Chart Agent exists. Placeholder for Phase 3.",
    },
    {
        "id": "T5-02",
        "tier": 5,
        "question": "Predict which customers are likely to churn based on order and ticket history.",
        "ground_truth": None,
        "answer_type": "model",
        "scoring_note": "Not scoreable until Model Agent exists. Placeholder for Phase 3.",
    },
]


def summary():
    from collections import Counter
    tier_counts = Counter(q["tier"] for q in BENCHMARK)
    print(f"Total benchmark questions: {len(BENCHMARK)}")
    for tier in sorted(tier_counts):
        print(f"  Tier {tier}: {tier_counts[tier]} questions")


if __name__ == "__main__":
    summary()
