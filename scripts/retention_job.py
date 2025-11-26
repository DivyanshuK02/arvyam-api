#!/usr/bin/env python3
# scripts/retention_job.py
"""
Phase 3.1 Retention Job — Daily De-Identification

Runs nightly (e.g., 02:00 IST via cron) to de-identify guest orders
older than 90 days.

Behavior:
- Finds orders older than 90 days where user_id IS NULL or user's memory_opt_in = FALSE
- Sets user_id = NULL, email = NULL (preserves SKU/amount for accounting)
- Does NOT delete orders (preserves financial records)
- Respects memory_opt_in flag (opted-in users exempt from de-identification)

Usage:
    # Dry run (no changes)
    python scripts/retention_job.py --dry-run
    
    # Execute de-identification
    python scripts/retention_job.py
    
    # With custom retention days
    python scripts/retention_job.py --days 60
    
    # Verbose output
    python scripts/retention_job.py --verbose

Cron example (02:00 IST daily):
    0 2 * * * /path/to/venv/bin/python /path/to/scripts/retention_job.py >> /var/log/retention.log 2>&1

Environment Variables Required:
    SUPABASE_URL: Database URL
    SUPABASE_KEY: Service role key
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("retention_job")


# ============================================================
# Constants
# ============================================================

DEFAULT_RETENTION_DAYS = 90
BATCH_SIZE = 100  # Process in batches for large datasets


# ============================================================
# Database Operations
# ============================================================

def get_client():
    """Get Supabase client."""
    try:
        from app.db import get_supabase_client
        return get_supabase_client()
    except ImportError:
        # Fallback: direct supabase import
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL", "").strip()
            key = os.getenv("SUPABASE_KEY", "").strip()
            if not url or not key:
                log.error("SUPABASE_URL and SUPABASE_KEY environment variables required")
                return None
            return create_client(url, key)
        except ImportError:
            log.error("supabase package not installed")
            return None


def find_orders_to_deidentify(
    client,
    retention_days: int,
    batch_size: int = BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Find orders eligible for de-identification.
    
    Criteria:
    - Created more than {retention_days} days ago
    - Either:
      - user_id IS NULL (guest order), OR
      - user's memory_opt_in = FALSE
    - email IS NOT NULL (not already de-identified)
    
    Args:
        client: Supabase client
        retention_days: Days after which to de-identify
        batch_size: Max orders to process per run
        
    Returns:
        List of order records eligible for de-identification
    """
    cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
    
    log.info(f"Finding orders older than {cutoff} (retention: {retention_days} days)")
    
    # Step 1: Find guest orders (user_id IS NULL) with email
    guest_orders_result = client.table("orders")\
        .select("id, email, user_id, created_at")\
        .is_("user_id", "null")\
        .not_.is_("email", "null")\
        .lt("created_at", cutoff)\
        .limit(batch_size)\
        .execute()
    
    guest_orders = guest_orders_result.data or []
    log.info(f"Found {len(guest_orders)} guest orders eligible for de-identification")
    
    # Step 2: Find orders where user has memory_opt_in = FALSE
    # This requires a join or subquery - we'll do it in two steps
    
    # Get users with memory_opt_in = FALSE
    non_opted_users_result = client.table("users")\
        .select("id")\
        .eq("memory_opt_in", False)\
        .execute()
    
    non_opted_user_ids = [u["id"] for u in (non_opted_users_result.data or [])]
    
    non_opted_orders = []
    if non_opted_user_ids:
        # Find orders for these users that are old enough
        for user_id in non_opted_user_ids[:50]:  # Limit to prevent huge queries
            orders_result = client.table("orders")\
                .select("id, email, user_id, created_at")\
                .eq("user_id", user_id)\
                .not_.is_("email", "null")\
                .lt("created_at", cutoff)\
                .limit(batch_size // len(non_opted_user_ids) + 1)\
                .execute()
            
            non_opted_orders.extend(orders_result.data or [])
    
    log.info(f"Found {len(non_opted_orders)} non-opted-in user orders eligible")
    
    # Combine and deduplicate
    all_orders = guest_orders + non_opted_orders
    seen_ids = set()
    unique_orders = []
    for order in all_orders:
        if order["id"] not in seen_ids:
            seen_ids.add(order["id"])
            unique_orders.append(order)
    
    return unique_orders[:batch_size]


def deidentify_orders(
    client,
    orders: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    De-identify orders by setting user_id and email to NULL.
    
    Args:
        client: Supabase client
        orders: List of orders to de-identify
        dry_run: If True, don't actually update
        
    Returns:
        Summary of operation
    """
    if not orders:
        return {"processed": 0, "success": 0, "failed": 0}
    
    processed = 0
    success = 0
    failed = 0
    
    for order in orders:
        order_id = order["id"]
        
        if dry_run:
            log.info(f"[DRY RUN] Would de-identify order {order_id}")
            processed += 1
            success += 1
            continue
        
        try:
            # Update: set user_id and email to NULL
            client.table("orders")\
                .update({"user_id": None, "email": None})\
                .eq("id", order_id)\
                .execute()
            
            success += 1
            log.debug(f"De-identified order {order_id}")
        except Exception as e:
            failed += 1
            log.error(f"Failed to de-identify order {order_id}: {e}")
        
        processed += 1
    
    return {"processed": processed, "success": success, "failed": failed}


# ============================================================
# Idempotency Guard
# ============================================================

def check_idempotency(client) -> bool:
    """
    Check if job has already run today (idempotency guard).
    
    Uses a simple marker in a jobs table or checks for recent runs.
    For MVP, we skip this check and rely on cron scheduling.
    
    Returns:
        True if safe to proceed, False if already ran today
    """
    # For MVP, always return True (rely on cron scheduling)
    # In production, could check a job_runs table
    return True


def record_job_run(client, summary: Dict[str, Any]) -> None:
    """
    Record job run for auditing and idempotency.
    
    In production, would insert into a job_runs table.
    For MVP, just logs the result.
    """
    log.info(f"Job completed: {summary}")


# ============================================================
# Main Entry Point
# ============================================================

def run_retention_job(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the retention/de-identification job.
    
    Args:
        retention_days: Days after which to de-identify
        dry_run: If True, don't actually make changes
        verbose: If True, enable debug logging
        
    Returns:
        Summary of job execution
    """
    if verbose:
        log.setLevel(logging.DEBUG)
    
    start_time = datetime.utcnow()
    log.info(f"=" * 60)
    log.info(f"RETENTION JOB STARTED")
    log.info(f"Retention period: {retention_days} days")
    log.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    log.info(f"=" * 60)
    
    # Get database client
    client = get_client()
    if not client:
        log.error("Failed to get database client")
        return {"status": "error", "message": "Database connection failed"}
    
    # Idempotency check
    if not check_idempotency(client):
        log.warning("Job already ran today, skipping")
        return {"status": "skipped", "message": "Already ran today"}
    
    try:
        # Find eligible orders
        orders = find_orders_to_deidentify(client, retention_days)
        log.info(f"Total orders to process: {len(orders)}")
        
        if not orders:
            log.info("No orders require de-identification")
            return {
                "status": "success",
                "processed": 0,
                "success": 0,
                "failed": 0,
                "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
            }
        
        # De-identify orders
        summary = deidentify_orders(client, orders, dry_run=dry_run)
        
        # Record job run
        summary["status"] = "success" if summary["failed"] == 0 else "partial"
        summary["duration_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        summary["retention_days"] = retention_days
        summary["dry_run"] = dry_run
        summary["timestamp"] = datetime.utcnow().isoformat()
        
        record_job_run(client, summary)
        
        log.info(f"=" * 60)
        log.info(f"RETENTION JOB COMPLETED")
        log.info(f"Processed: {summary['processed']}")
        log.info(f"Success: {summary['success']}")
        log.info(f"Failed: {summary['failed']}")
        log.info(f"Duration: {summary['duration_ms']}ms")
        log.info(f"=" * 60)
        
        return summary
        
    except Exception as e:
        log.error(f"Job failed with error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3.1 Retention Job - De-identify old guest orders"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Retention period in days (default: {DEFAULT_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) output",
    )
    
    args = parser.parse_args()
    
    summary = run_retention_job(
        retention_days=args.days,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Exit with appropriate code
    if summary.get("status") == "error":
        sys.exit(1)
    elif summary.get("failed", 0) > 0:
        sys.exit(2)  # Partial failure
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
