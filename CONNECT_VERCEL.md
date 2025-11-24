# Connect GitHub Repository to Vercel

## Step-by-Step Instructions

### Step 1: Go to Vercel Dashboard
1. Open https://vercel.com/dashboard
2. Make sure you're logged in

### Step 2: Add New Project
1. Click the **"Add New..."** button (usually top right or center)
2. Select **"Project"** from the dropdown

### Step 3: Import Git Repository
1. You'll see a page asking to "Import Git Repository"
2. If you see GitHub repositories listed:
   - Look for `kavya5jan-prog/autoclaim`
   - Click **"Import"** next to it
3. If you don't see your repository:
   - Click **"Adjust GitHub App Permissions"** or **"Configure GitHub App"**
   - Or click **"Import Git Repository"** → **"GitHub"**
   - You may need to authorize Vercel to access your GitHub account

### Step 4: Authorize Vercel (if needed)
1. If prompted, click **"Authorize Vercel"** or **"Install"**
2. Select which repositories to give access:
   - Choose **"All repositories"** OR
   - Select **"Only select repositories"** → choose `autoclaim`
3. Click **"Install"** or **"Authorize"**
4. You'll be redirected back to Vercel

### Step 5: Select Your Repository
1. After authorization, you should see `kavya5jan-prog/autoclaim` in the list
2. Click **"Import"** next to it

### Step 6: Configure Project
1. **Project Name**: `autoclaim` (or leave default)
2. **Framework Preset**: Select **"Other"** or **"No Framework"**
3. **Root Directory**: `./` (default - leave as is)
4. **Build Command**: Leave **empty**
5. **Output Directory**: Leave **empty**
6. **Install Command**: `pip install -r requirements.txt`

### Step 7: Add Environment Variables
1. Scroll down to **"Environment Variables"** section
2. Click **"Add"** or **"Add Environment Variable"**
3. Add:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: (paste your OpenAI API key)
   - **Environments**: Select all three:
     - ☑ Production
     - ☑ Preview  
     - ☑ Development
4. Click **"Save"** or **"Add"**

### Step 8: Deploy
1. Scroll down and click **"Deploy"** button
2. Wait for the deployment to complete (usually 2-5 minutes)
3. You'll see build logs in real-time

### Step 9: Access Your App
1. Once deployment completes, you'll see:
   - ✅ "Ready" status
   - A URL like: `https://autoclaim.vercel.app`
2. Click the URL to open your app!

## Troubleshooting

### "Repository not found" or "Access denied"
- Make sure the repository is **Public** OR
- Make sure Vercel has access to your **Private** repositories
- Check GitHub Settings → Applications → Authorized OAuth Apps → Vercel

### Can't see the repository in the list
1. Click **"Adjust GitHub App Permissions"**
2. Make sure Vercel has access to the repository
3. Try refreshing the page

### Still having issues?
Try this alternative:
1. Go to https://vercel.com/new
2. Select "Import Git Repository"
3. Search for: `kavya5jan-prog/autoclaim`
4. Follow the steps above

## After Connection

Once connected, Vercel will:
- ✅ Automatically deploy when you push to `main` branch
- ✅ Create preview deployments for pull requests
- ✅ Show all deployments in the dashboard

Your repository URL: https://github.com/kavya5jan-prog/autoclaim

