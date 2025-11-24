#!/bin/bash
# Quick deployment script for Vercel

echo "🚀 Deploying to Vercel..."

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Login (if not already)
echo "🔐 Checking Vercel login..."
vercel whoami || vercel login

# Deploy
echo "📤 Deploying project..."
vercel --prod

echo "✅ Deployment complete!"
echo "💡 Don't forget to set OPENAI_API_KEY environment variable in Vercel dashboard!"

