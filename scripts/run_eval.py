#!/usr/bin/env python3
"""Standalone CI/CD evaluation runner.

Usage:
    python scripts/run_eval.py --clients acme_corp techstart --k 5
    # Exit code 0 = pass, 1 = fail
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Multi Tenant RAG evaluation runner")
    parser.add_argument("--clients", nargs="+", default=["acme_corp", "techstart"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Default test queries per client
    default_queries = {
        "acme_corp": [
            "What is the annual leave entitlement?",
            "How do performance reviews work?",
            "What are the code review requirements?",
            "What is the incident response time for severity 1?",
        ],
        "techstart": [
            "How do I reset my password?",
            "What integrations does TechStart support?",
            "What are the pricing plans?",
            "How does the cash flow forecast work?",
        ],
    }

    from app.clients.manager import register
    from app.evaluation.pipeline import run_eval_pipeline, EvalQuery

    # Ensure clients exist (register if not)
    for cid in args.clients:
        register(client_id=cid, display_name=cid)

    client_queries = {
        cid: [EvalQuery(query=q, relevant_chunk_ids=[]) for q in default_queries.get(cid, [])]
        for cid in args.clients
        if cid in default_queries
    }

    if not client_queries:
        print("No evaluation queries configured for the specified clients.")
        sys.exit(0)

    print(f"Running evaluation for clients: {', '.join(client_queries.keys())}")
    print(f"Rank cut-off k={args.k}\n")

    report = run_eval_pipeline(client_queries, k=args.k)

    for result in report.client_results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} | {result.client_id}")
        for key, val in result.metrics.items():
            bar = "█" * int(val * 20)
            print(f"       {key:15s} {val:.3f} {bar}")
        if result.failure_reasons:
            for reason in result.failure_reasons:
                print(f"       ↳ {reason}")
        print()

    print(f"Overall: {'PASSED' if report.passed else 'FAILED'} "
          f"({report.passed_clients}/{report.total_clients} clients passed)")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
