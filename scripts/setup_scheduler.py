#!/usr/bin/env python3
"""
Setup Cloud Scheduler to trigger market data ingestion every hour.
"""

import argparse
import subprocess
import sys


def create_scheduler_job(project_id: str, function_name: str = "ingest-market-data"):
    """Create Cloud Scheduler job to trigger the Cloud Function."""
    
    job_name = "market-data-ingestion"
    schedule = "0 * * * *"  # Every hour at minute 0
    timezone = "UTC"
    
    # Get the function URL
    try:
        result = subprocess.run([
            "gcloud", "functions", "describe", function_name,
            "--project", project_id,
            "--format", "value(httpsTrigger.url)"
        ], capture_output=True, text=True, check=True)
        
        function_url = result.stdout.strip()
        if not function_url:
            raise ValueError("Could not get function URL")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting function URL: {e}")
        print("Make sure the Cloud Function is deployed first:")
        print(f"gcloud functions deploy {function_name} --source cloud_functions/ --runtime python311 --trigger-http --allow-unauthenticated")
        sys.exit(1)
    
    # Create the scheduler job
    cmd = [
        "gcloud", "scheduler", "jobs", "create", "http", job_name,
        "--project", project_id,
        "--schedule", schedule,
        "--time-zone", timezone,
        "--uri", function_url,
        "--http-method", "POST",
        "--headers", "Content-Type=application/json",
        "--message-body", '{"test": "scheduled"}',
        "--replace"  # Replace if exists
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Created scheduler job '{job_name}'")
        print(f"   Schedule: {schedule} ({timezone})")
        print(f"   Function: {function_url}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating scheduler job: {e}")
        sys.exit(1)


def test_scheduler_job(project_id: str, job_name: str = "market-data-ingestion"):
    """Test the scheduler job by running it once."""
    
    cmd = [
        "gcloud", "scheduler", "jobs", "run", job_name,
        "--project", project_id
    ]
    
    try:
        print(f"Running scheduler job '{job_name}'...")
        subprocess.run(cmd, check=True)
        print("✅ Scheduler job executed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running scheduler job: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Setup Cloud Scheduler for market data ingestion")
    parser.add_argument("project_id", help="GCP Project ID")
    parser.add_argument("--function-name", default="ingest-market-data", help="Cloud Function name")
    parser.add_argument("--test", action="store_true", help="Test the scheduler job")
    args = parser.parse_args()
    
    if args.test:
        test_scheduler_job(args.project_id)
    else:
        create_scheduler_job(args.project_id, args.function_name)
        print("\n🎉 Scheduler setup complete!")
        print("The function will run every hour automatically.")
        print(f"To test: python scripts/setup_scheduler.py {args.project_id} --test")


if __name__ == "__main__":
    main()
