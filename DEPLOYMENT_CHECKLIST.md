# 📋 Deployment Checklist

Use this checklist before deploying to production!

## Pre-Deployment

### Code Quality
- [ ] All files committed to Git
- [ ] `.gitignore` configured properly
- [ ] No sensitive data in code (API keys, passwords)
- [ ] requirements.txt up to date

### Model Files
- [ ] Model trained successfully (`python train_model.py`)
- [ ] All `.pkl` files exist in `models/` directory:
  - [ ] `fraud_detector.pkl`
  - [ ] `scaler.pkl`
  - [ ] `feature_names.pkl`
  - [ ] `metadata.pkl`
- [ ] Model files committed to Git (if <100MB each)

### Django Settings
- [ ] `SECRET_KEY` set to environment variable
- [ ] `DEBUG = False` in production
- [ ] `ALLOWED_HOSTS` configured for your domain
- [ ] Static files configuration tested
- [ ] Database configured (SQLite for small scale, PostgreSQL for production)

### Testing
- [ ] Tested locally with `python manage.py runserver`
- [ ] Prediction form works correctly
- [ ] API endpoint returns valid JSON
- [ ] Model info page displays correctly
- [ ] Error handling tested

## Deployment to Render

### Initial Setup
- [ ] Render account created
- [ ] GitHub repository connected
- [ ] Web service created
- [ ] Build command: `./build.sh`
- [ ] Start command: `gunicorn fraud_detection.wsgi:application`

### Environment Variables
- [ ] `SECRET_KEY` set (auto-generated or custom)
- [ ] `DEBUG=False` set
- [ ] `PYTHON_VERSION=3.11.0` set

### Post-Deployment
- [ ] Service deployed successfully
- [ ] URL accessible (e.g., https://your-app.onrender.com)
- [ ] Prediction form works
- [ ] API endpoint functional
- [ ] Static files loading correctly
- [ ] No errors in logs

## Performance Optimization

### Optional Enhancements
- [ ] Upgrade to paid Render plan (for always-on service)
- [ ] Add custom domain
- [ ] Configure CDN for static files
- [ ] Set up monitoring/logging
- [ ] Add rate limiting
- [ ] Implement caching

## Security Hardening

- [ ] Change default admin credentials
- [ ] Enable HTTPS only
- [ ] Add CORS headers if needed
- [ ] Implement rate limiting
- [ ] Add request logging
- [ ] Regular security updates

## Monitoring

- [ ] Set up error tracking (Sentry, etc.)
- [ ] Monitor prediction latency
- [ ] Track API usage
- [ ] Monitor model performance drift

---

## Quick Deploy Commands

```bash
# 1. Ensure model is trained
python train_model.py

# 2. Test locally
python manage.py runserver

# 3. Commit and push
git add .
git commit -m "Ready for deployment"
git push origin main

# 4. Deploy on Render
# Follow the web UI steps or use Render CLI
```

---

## Rollback Plan

If deployment fails:

1. Check Render logs for errors
2. Verify all environment variables
3. Ensure model files are present
4. Test locally to reproduce issue
5. Fix and redeploy

---

**Last Updated:** February 2025  
**Project:** Fraud Detection System
