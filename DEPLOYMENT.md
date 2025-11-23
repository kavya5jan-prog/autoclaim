# Deployment Guide

This guide covers deploying the Auto Claims Analysis application to cloud platforms.

## Prerequisites

- Git repository (GitHub, GitLab, or Bitbucket)
- OpenAI API key
- Account on a cloud platform (Render.com recommended)

## Environment Variables

The application requires the following environment variable:

- `OPENAI_API_KEY` - Your OpenAI API key for GPT-4o access

## Deploy to Render.com (Recommended)

### Step 1: Push to GitHub

1. Initialize git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a repository on GitHub and push:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. Go to [Render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: auto-claims (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or choose a paid plan)

5. Add Environment Variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

6. Click "Create Web Service"
7. Wait for deployment to complete (usually 2-5 minutes)

Your app will be available at: `https://your-app-name.onrender.com`

## Deploy to Railway

### Step 1: Install Railway CLI (optional)

```bash
npm i -g @railway/cli
railway login
```

### Step 2: Deploy

1. Go to [Railway.app](https://railway.app) and sign up/login
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add environment variable:
   - `OPENAI_API_KEY` = your API key
5. Railway will automatically detect the Procfile and deploy

## Deploy to Fly.io

### Step 1: Install Fly CLI

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### Step 2: Create fly.toml

Create a `fly.toml` file:
```toml
app = "your-app-name"
primary_region = "iad"

[build]

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
```

### Step 3: Deploy

```bash
fly launch
fly secrets set OPENAI_API_KEY=your-api-key
fly deploy
```

## Deploy to Heroku

### Step 1: Install Heroku CLI

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Or download from https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy

```bash
heroku login
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-api-key
git push heroku main
```

## Local Production Testing

Test the production setup locally:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app:app --bind 0.0.0.0:5001
```

## Troubleshooting

### Port Issues
- Render and Railway automatically set the `PORT` environment variable
- The app is configured to use `PORT` if available, otherwise defaults to 5001

### OpenAI API Errors
- Ensure `OPENAI_API_KEY` is set correctly in your platform's environment variables
- Check that your OpenAI account has sufficient credits

### Build Failures
- Ensure all dependencies are in `requirements.txt`
- Check that Python version is compatible (3.9+)

### File Upload Issues
- The app allows up to 16MB file uploads
- Ensure your platform allows sufficient request timeout for large files

## Notes

- Free tiers may have cold start delays (Render: ~50 seconds, Railway: ~30 seconds)
- For production use, consider upgrading to a paid plan for better performance
- The `uploads/` directory is ephemeral on most platforms - files are deleted after processing

