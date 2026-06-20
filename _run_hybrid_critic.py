"""Phase 2 benchmark with hybrid Critic (schema-derived + domain knowledge layers)."""
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "evals"))

from agents.schema_agent import get_schema, get_schema_with_enums
from agents.sql_agent import ask
from scoring import run_benchmark

client = anthropic.Anthropic()
schema = get_schema("data/analytics.db")
schema_with_enums = get_schema_with_enums("data/analytics.db")
db = "data/analytics.db"
fn = lambda q: ask(q, schema, schema_with_enums, db, client)

for label in ["phase2_hybrid_critic_run1", "phase2_hybrid_critic_run2", "phase2_hybrid_critic_run3"]:
    run_benchmark(fn, label, tiers=[1, 2, 3])
    print()
