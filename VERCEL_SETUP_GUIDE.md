# Vercel Setup Guide - Step by Step

## Issue: No Deployments Showing

If you see "No Production Deployment" and no deployments in the list, follow these steps:

## Step 1: Verify GitHub Repository Connection

1. In Vercel dashboard, click **"Repository"** button (top right, GitHub icon)
2. You should see your repository: `kavya5jan-prog/autoclaim`
3. If you see "Connect Git Repository" instead:
   - Click "Connect Git Repository"
   - Select GitHub
   - Authorize Vercel if needed
   - Search for `autoclaim` or `kavya5jan-prog/autoclaim`
   - Click "Import"

## Step 2: Create New Deployment Manually

If the repository is connected but no deployments exist:

1. Go to your project dashboard
2. Click **"Deployments"** tab (bottom right)
3. Click **"Create Deployment"** or **"Deploy"** button
4. Select:
   - **Branch**: `main`
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`

## Step 3: Add Environment Variables

**CRITICAL**: Before deploying, add environment variables:

1. Go to **Project Settings** → **Environment Variables**
2. Click **"Add New"**
3. Add:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key
   - **Environment**: Select all (Production, Preview, Development)
4. Click **"Save"**

## Step 4: Trigger Deployment

After setting up:

### Option A: Via Dashboard
1. Go to **"Deployments"** tab
2. Click **"Redeploy"** or **"Deploy"**
3. Select `main` branch
4. Click **"Deploy"**

### Option B: Via Git Push (if auto-deploy is enabled)
```bash
# Make a small change to trigger deployment
echo "" >> README.md
git add README.md
git commit -m "Trigger deployment"
git push origin main
```

## Step 5: Check Build Status

After triggering deployment:

1. Go to **"Deployments"** tab
2. You should see a new deployment with status:
   - 🟡 Building
   - 🟢 Ready
   - 🔴 Error

3. Click on the deployment to see:
   - Build logs
   - Function logs
   - Any errors

## Troubleshooting

### If "Connect Git Repository" doesn't work:
- Make sure you're logged into GitHub in the same browser
- Check that Vercel has access to your repositories
- Go to GitHub Settings → Applications → Authorized OAuth Apps → Vercel

### If deployment fails:
- Check build logs for errors
- Common issues:
  - Missing dependencies in requirements.txt
  - Python version mismatch
  - Import errors
  - Memory/timeout limits

### If no deployments appear at all:
- Try creating a new project:
  1. Click "Add New..." → "Project"
  2. Import `kavya5jan-prog/autoclaim`
  3. Configure as above
  4. Deploy

## Quick Test: Create a Simple Deployment

If nothing works, let's verify the setup with a minimal test:

1. In Vercel dashboard, click **"Add New..."** → **"Project"**
2. Import: `kavya5jan-prog/autoclaim`
3. Configure:
   - Framework: Other
   - Root: `./`
   - Build: (empty)
   - Output: (empty)
   - Install: `pip install -r requirements.txt`
4. Add Environment Variable: `OPENAI_API_KEY`
5. Click **"Deploy"**

This should create your first deployment!

