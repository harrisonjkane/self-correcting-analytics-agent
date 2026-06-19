"""
Builds a realistic multi-table SQLite test database for AI Analytics Consultant.

Schema: customers, products, orders, order_items, support_tickets
Designed specifically to exercise multi-table joins, aggregation, and
ambiguous business-logic questions for the evaluation benchmark.
"""

import sqlite3
import random
from datetime import datetime, timedelta
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "analytics.db")

random.seed(42)  # reproducible data across rebuilds

# ── Reference data ──────────────────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "Central"]
CUSTOMER_SEGMENTS = ["Enterprise", "SMB", "Startup"]
PRODUCT_CATEGORIES = ["Software", "Hardware", "Services", "Support Plans"]
TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]
TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
ORDER_STATUSES = ["Completed", "Pending", "Cancelled", "Refunded"]


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Schema ───────────────────────────────────────────────────────────
    cur.executescript("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        company_name TEXT NOT NULL,
        segment TEXT NOT NULL,
        region TEXT NOT NULL,
        signup_date TEXT NOT NULL,
        account_owner TEXT NOT NULL
    );

    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );

    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );

    CREATE TABLE support_tickets (
        ticket_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        opened_date TEXT NOT NULL,
        closed_date TEXT,
        priority TEXT NOT NULL,
        status TEXT NOT NULL,
        subject TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """)

    # ── Customers (60) ──────────────────────────────────────────────────
    company_prefixes = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne",
                         "Wonka", "Hooli", "Pied Piper", "Massive Dynamic", "Cyberdyne",
                         "Soylent", "Tyrell", "Aperture", "Black Mesa", "Oscorp",
                         "Vandelay", "Gekko & Co", "Dunder Mifflin", "Prestige Worldwide"]
    company_suffixes = ["Inc", "LLC", "Group", "Holdings", "Partners", "Co"]

    owners = ["J. Martinez", "A. Chen", "R. Patel", "S. Johnson", "K. Williams", "M. Garcia"]

    customers = []
    start_date = datetime(2021, 1, 1)
    for i in range(1, 61):
        prefix = random.choice(company_prefixes)
        suffix = random.choice(company_suffixes)
        signup = start_date + timedelta(days=random.randint(0, 1400))
        customers.append((
            i,
            f"{prefix} {suffix} {i}",
            random.choices(CUSTOMER_SEGMENTS, weights=[0.25, 0.45, 0.30])[0],
            random.choice(REGIONS),
            signup.strftime("%Y-%m-%d"),
            random.choice(owners),
        ))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    # ── Products (18) ───────────────────────────────────────────────────
    products = [
        (1, "Core Platform License", "Software", 4999.00),
        (2, "Analytics Add-on", "Software", 1499.00),
        (3, "API Access Tier", "Software", 799.00),
        (4, "Mobile App License", "Software", 999.00),
        (5, "Enterprise SSO", "Software", 2499.00),
        (6, "Edge Server Unit", "Hardware", 3200.00),
        (7, "Network Appliance", "Hardware", 5400.00),
        (8, "Backup Storage Array", "Hardware", 2800.00),
        (9, "Onboarding Package", "Services", 2000.00),
        (10, "Custom Integration", "Services", 6500.00),
        (11, "Data Migration", "Services", 3500.00),
        (12, "Training Workshop", "Services", 1200.00),
        (13, "Basic Support Plan", "Support Plans", 600.00),
        (14, "Premium Support Plan", "Support Plans", 1800.00),
        (15, "24/7 Critical Support", "Support Plans", 4200.00),
        (16, "Security Audit", "Services", 4800.00),
        (17, "Reporting Module", "Software", 899.00),
        (18, "Compliance Toolkit", "Software", 1699.00),
    ]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    # ── Orders + order_items ────────────────────────────────────────────
    order_id = 1
    order_item_id = 1
    orders_rows = []
    order_items_rows = []

    for cust_id, *_rest in customers:
        signup = datetime.strptime(customers[cust_id - 1][4], "%Y-%m-%d")
        num_orders = random.randint(1, 8)
        for _ in range(num_orders):
            order_date = signup + timedelta(days=random.randint(0, 700))
            if order_date > datetime(2024, 12, 31):
                order_date = datetime(2024, 12, 31) - timedelta(days=random.randint(0, 200))
            status = random.choices(ORDER_STATUSES, weights=[0.78, 0.08, 0.07, 0.07])[0]
            orders_rows.append((order_id, cust_id, order_date.strftime("%Y-%m-%d"), status))

            num_items = random.randint(1, 4)
            chosen_products = random.sample(products, num_items)
            for prod in chosen_products:
                qty = random.randint(1, 5)
                order_items_rows.append((
                    order_item_id, order_id, prod[0], qty, prod[3]
                ))
                order_item_id += 1
            order_id += 1

    cur.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders_rows)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", order_items_rows)

    # ── Support tickets ─────────────────────────────────────────────────
    ticket_id = 1
    ticket_rows = []
    subjects = [
        "Login authentication failure", "API rate limit question", "Billing discrepancy",
        "Feature request: bulk export", "Performance degradation on dashboard",
        "Integration with third-party CRM", "Data sync delay", "SSO configuration help",
        "Mobile app crash on startup", "Report generation timeout",
        "Onboarding training request", "Security audit follow-up",
        "Storage capacity question", "User permission escalation",
        "Unexpected downtime report",
    ]
    for cust_id, *_rest in customers:
        num_tickets = random.randint(0, 6)
        signup = datetime.strptime(customers[cust_id - 1][4], "%Y-%m-%d")
        for _ in range(num_tickets):
            opened = signup + timedelta(days=random.randint(10, 800))
            if opened > datetime(2024, 12, 31):
                opened = datetime(2024, 12, 31) - timedelta(days=random.randint(0, 150))
            status = random.choices(TICKET_STATUSES, weights=[0.10, 0.10, 0.20, 0.60])[0]
            closed = None
            if status in ("Resolved", "Closed"):
                closed = (opened + timedelta(days=random.randint(1, 21))).strftime("%Y-%m-%d")
            ticket_rows.append((
                ticket_id, cust_id, opened.strftime("%Y-%m-%d"), closed,
                random.choices(TICKET_PRIORITIES, weights=[0.35, 0.35, 0.22, 0.08])[0],
                status, random.choice(subjects),
            ))
            ticket_id += 1

    cur.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", ticket_rows)

    conn.commit()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"Database built at {DB_PATH}")
    for table in ["customers", "products", "orders", "order_items", "support_tickets"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    build()
