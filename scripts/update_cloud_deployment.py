#!/usr/bin/env python3
"""
Update Cloud Deployment Script
Updates the deployed cloud services with latest code changes.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, description=""):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✅ Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.stderr}")
        raise

def update_service(service_name, image_tag="latest"):
    """Update a specific Cloud Run service"""

    project_id = "quant-ai-trader-credits"
    region = "us-central1"

    # Build new image
    image_name = f"gcr.io/{project_id}/{service_name}:{image_tag}"
    run_command(
        ["docker", "build", "-t", image_name, "."],
        f"Building Docker image for {service_name}"
    )

    # Push image
    run_command(
        ["docker", "push", image_name],
        f"Pushing image {image_name}"
    )

    # Deploy to Cloud Run
    run_command([
        "gcloud", "run", "deploy", service_name,
        f"--image={image_name}",
        "--platform=managed",
        f"--region={region}",
        f"--project={project_id}",
        "--allow-unauthenticated"
    ], f"Deploying {service_name} to Cloud Run")

def main():
    parser = argparse.ArgumentParser(description="Update Cloud Run deployments")
    parser.add_argument("--service", help="Specific service to update")
    parser.add_argument("--all", action="store_true", help="Update all services")
    parser.add_argument("--tag", default="latest", help="Docker image tag")

    args = parser.parse_args()

    services = [
        "aster-trading-agent",
        "aster-self-learning-trader",
        "aster-enhanced-dashboard",
        "aster-modern-frontend"
    ]

    if args.service:
        if args.service not in services:
            print(f"❌ Unknown service: {args.service}")
            print(f"   Available services: {', '.join(services)}")
            sys.exit(1)
        services = [args.service]
    elif not args.all:
        print("Please specify --service <name> or --all")
        sys.exit(1)

    print("🚀 Updating AsterAI Cloud Deployments")
    print("=" * 50)

    for service in services:
        try:
            print(f"\n📦 Updating {service}...")
            update_service(service, args.tag)
            print(f"✅ {service} updated successfully")
        except Exception as e:
            print(f"❌ Failed to update {service}: {e}")
            continue

    print("\n🎉 Deployment update completed!")
    print("\n📋 Check status with:")
    print("   python3 scripts/start_cloud_bots.py --action status")

if __name__ == "__main__":
    main()
