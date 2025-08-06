# Google Drive Integration Setup Guide

This guide will help you set up Google Drive integration for the Automata Learning Tool.

## Overview

The Google Drive integration allows users to:
- Save their learning progress to their own Google Drive
- Sync progress across devices
- Keep their data private (we never see it)
- Export/import progress anytime

## Setup Steps

### 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Name it something like "Automata Learning Tool"

### 2. Enable Google Drive API

1. In your project, go to "APIs & Services" > "Library"
2. Search for "Google Drive API"
3. Click on it and press "Enable"

### 3. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in the required fields:
     - App name: "Automata Learning Tool"
     - User support email: Your email
     - Developer contact: Your email
   - Add scopes: `https://www.googleapis.com/auth/drive.file`
   - Add test users if in development

4. Back in "Create OAuth client ID":
   - Application type: "Web application"
   - Name: "Automata Web Client"
   - Authorized JavaScript origins:
     - `http://localhost:3000` (development)
     - `http://localhost:5173` (Vite default)
     - `https://your-domain.com` (production)
   - No redirect URIs needed (we use implicit flow)

5. Click "Create" and save your Client ID

### 4. Create API Key (Optional)

1. Click "Create Credentials" > "API key"
2. Restrict the key:
   - Application restrictions: "HTTP referrers"
   - Add your domains
   - API restrictions: "Google Drive API"

### 5. Configure the Frontend

Create a `.env` file in the frontend directory:

```bash
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
VITE_GOOGLE_API_KEY=your-api-key  # Optional
```

### 6. Test the Integration

1. Start the development server
2. Complete a few problems
3. Click "Save to Google Drive" when prompted
4. Sign in with your Google account
5. Check your Google Drive for the "Automata Progress" folder

## Privacy & Security Notes

### What We Access
- **Scope**: `drive.file` - Only files created by this app
- **Cannot**: Access any other files in user's Drive
- **Cannot**: Delete or modify files outside our folder

### Data Flow
```
Browser → Google OAuth → Google Drive
   ↓                         ↑
   └─────── Direct ──────────┘
   
(Never touches our servers)
```

### User Control
- Users can revoke access anytime at [Google Account Permissions](https://myaccount.google.com/permissions)
- Users can delete the "Automata Progress" folder anytime
- Users can export their data as JSON

## Troubleshooting

### "Unauthorized" Error
- Check that your domain is in the authorized origins
- Ensure the Google Drive API is enabled
- Verify the client ID is correct

### "Scope not authorized"
- Make sure you've added the `drive.file` scope in OAuth consent screen
- Users may need to re-authenticate after scope changes

### Rate Limits
- Google Drive API has generous limits (1 billion requests/day)
- We only save when user clicks save or at milestones
- Each user's quota is separate

## Production Checklist

- [ ] Move OAuth consent screen from "Testing" to "Production"
- [ ] Add production domain to authorized origins
- [ ] Implement proper error handling for offline scenarios
- [ ] Add privacy policy mentioning Google Drive usage
- [ ] Test on multiple browsers and devices

## Alternative: Local Storage Only

If you prefer not to use Google Drive:
1. Remove the Google Drive components
2. Use only localStorage for persistence
3. Add export/import functionality for manual backups

## Support

For issues with Google APIs, see [Google Drive API Documentation](https://developers.google.com/drive/api/v3/about-sdk)

For app-specific issues, please open an issue on GitHub.