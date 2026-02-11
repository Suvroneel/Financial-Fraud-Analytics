# 🚀 Quick Start Guide

Get your fraud detection system running in **5 minutes**!

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] Fraud.csv dataset downloaded

## Step-by-Step Setup

### 1️⃣ Clone & Navigate
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Dataset
Place `Fraud.csv` in the project root directory:
```
fraud-detection-system/
├── Fraud.csv          <-- Put it here!
├── train_model.py
├── manage.py
└── ...
```

### 4️⃣ Train Model (~8 minutes)
```bash
python train_model.py
```

Wait for this output:
```
✓ TRAINING COMPLETE - All artifacts saved successfully!
```

### 5️⃣ Run Django App
```bash
python manage.py runserver
```

### 6️⃣ Open Browser
Navigate to: **http://localhost:8000**

## 🎉 You're Done!

Try a sample transaction:
- Amount: `250000`
- Type: `CASH_OUT`
- Origin Old Balance: `300000`
- Origin New Balance: `50000`
- Destination Old Balance: `0`
- Destination New Balance: `250000`

This should detect as **FRAUD** with high probability!

## 🐛 Troubleshooting

**Problem:** Model files not found  
**Solution:** Run `python train_model.py` first

**Problem:** Port 8000 already in use  
**Solution:** Use `python manage.py runserver 8001`

**Problem:** Missing dependencies  
**Solution:** Run `pip install -r requirements.txt` again

## 📦 Deploy to Render

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new Web Service
4. Connect your repository
5. Deploy!

Full instructions in [README.md](README.md#deployment)

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Email: suvroneelnathak213@gmail.com

---

**Happy Fraud Hunting! 🛡️**
