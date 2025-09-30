#!/bin/bash

echo "🧠 Deploying ExoBengal API to Cerebrium..."

# Check if cerebrium is installed
if ! command -v cerebrium &> /dev/null; then
    echo "Installing Cerebrium CLI..."
    pip install cerebrium
fi

# Login check
echo "Checking Cerebrium login status..."
if ! cerebrium whoami &> /dev/null; then
    echo "Please login to Cerebrium first:"
    echo "cerebrium login"
    exit 1
fi

# Deploy
echo "Deploying to Cerebrium..."
cerebrium deploy

echo "✅ Deployment complete!"
echo "Check your deployment at: https://dashboard.cerebrium.ai"
