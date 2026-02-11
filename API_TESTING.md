# 🧪 API Testing Guide

Test your fraud detection API using these examples!

## Base URL

**Local:** `http://localhost:8000`  
**Production:** `https://your-app.onrender.com`

---

## API Endpoints

### 1. Predict Fraud

**Endpoint:** `POST /api/predict/`  
**Content-Type:** `application/json`

#### Example 1: High-Risk Transaction (Should Detect Fraud)

```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250000.00,
    "oldbalanceOrg": 300000.00,
    "newbalanceOrig": 50000.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 250000.00,
    "type": "CASH_OUT"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "prediction": {
    "is_fraud": true,
    "fraud_probability": 0.8734,
    "confidence": 0.8734
  }
}
```

#### Example 2: Low-Risk Transaction (Should Be Legitimate)

```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.00,
    "oldbalanceOrg": 50000.00,
    "newbalanceOrig": 45000.00,
    "oldbalanceDest": 10000.00,
    "newbalanceDest": 15000.00,
    "type": "PAYMENT"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "prediction": {
    "is_fraud": false,
    "fraud_probability": 0.0234,
    "confidence": 0.9766
  }
}
```

#### Example 3: TRANSFER Transaction

```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100000.00,
    "oldbalanceOrg": 150000.00,
    "newbalanceOrig": 50000.00,
    "oldbalanceDest": 20000.00,
    "newbalanceDest": 120000.00,
    "type": "TRANSFER"
  }'
```

---

## Python Examples

### Using `requests` library

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/predict/"

# Transaction data
data = {
    "amount": 250000.00,
    "oldbalanceOrg": 300000.00,
    "newbalanceOrig": 50000.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 250000.00,
    "type": "CASH_OUT"
}

# Make request
response = requests.post(url, json=data)

# Parse response
result = response.json()

print(f"Is Fraud: {result['prediction']['is_fraud']}")
print(f"Probability: {result['prediction']['fraud_probability']:.2%}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

### Batch Processing Multiple Transactions

```python
import requests

url = "http://localhost:8000/api/predict/"

transactions = [
    {
        "amount": 250000.00,
        "oldbalanceOrg": 300000.00,
        "newbalanceOrig": 50000.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 250000.00,
        "type": "CASH_OUT"
    },
    {
        "amount": 5000.00,
        "oldbalanceOrg": 50000.00,
        "newbalanceOrig": 45000.00,
        "oldbalanceDest": 10000.00,
        "newbalanceDest": 15000.00,
        "type": "PAYMENT"
    }
]

results = []
for txn in transactions:
    response = requests.post(url, json=txn)
    results.append(response.json())

# Print results
for i, result in enumerate(results):
    print(f"Transaction {i+1}:")
    print(f"  Fraud: {result['prediction']['is_fraud']}")
    print(f"  Probability: {result['prediction']['fraud_probability']:.4f}")
    print()
```

---

## JavaScript/Node.js Example

```javascript
const fetch = require('node-fetch');

const url = 'http://localhost:8000/api/predict/';

const data = {
  amount: 250000.00,
  oldbalanceOrg: 300000.00,
  newbalanceOrig: 50000.00,
  oldbalanceDest: 0.00,
  newbalanceDest: 250000.00,
  type: 'CASH_OUT'
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
  console.log('Is Fraud:', result.prediction.is_fraud);
  console.log('Probability:', result.prediction.fraud_probability);
  console.log('Confidence:', result.prediction.confidence);
})
.catch(error => {
  console.error('Error:', error);
});
```

---

## Error Handling

### Missing Fields

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250000.00,
    "type": "CASH_OUT"
  }'
```

**Response:**
```json
{
  "error": "Missing required field: oldbalanceOrg"
}
```

### Invalid Transaction Type

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250000.00,
    "oldbalanceOrg": 300000.00,
    "newbalanceOrig": 50000.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 250000.00,
    "type": "INVALID_TYPE"
  }'
```

**Response:**
```json
{
  "error": "Invalid transaction type"
}
```

---

## Transaction Types

Valid transaction types:
- `CASH_OUT` - Cash withdrawal (High fraud risk)
- `TRANSFER` - Transfer between accounts (High fraud risk)
- `PAYMENT` - Payment transaction (Low fraud risk)
- `CASH_IN` - Cash deposit (Low fraud risk)
- `DEBIT` - Debit transaction (Low fraud risk)

---

## Performance Testing

### Load Testing with Apache Bench

```bash
# Test 100 requests with 10 concurrent connections
ab -n 100 -c 10 -p test_transaction.json -T application/json \
   http://localhost:8000/api/predict/
```

Where `test_transaction.json`:
```json
{
  "amount": 250000.00,
  "oldbalanceOrg": 300000.00,
  "newbalanceOrig": 50000.00,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 250000.00,
  "type": "CASH_OUT"
}
```

---

## Monitoring

### Check Prediction Latency

```python
import requests
import time

url = "http://localhost:8000/api/predict/"
data = {
    "amount": 250000.00,
    "oldbalanceOrg": 300000.00,
    "newbalanceOrig": 50000.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 250000.00,
    "type": "CASH_OUT"
}

start = time.time()
response = requests.post(url, json=data)
latency = (time.time() - start) * 1000  # Convert to ms

print(f"Latency: {latency:.2f}ms")
print(f"Status Code: {response.status_code}")
```

**Expected latency:** <100ms on local machine

---

## Troubleshooting

**Problem:** Connection refused  
**Solution:** Make sure Django server is running (`python manage.py runserver`)

**Problem:** 500 Internal Server Error  
**Solution:** Check that model files exist in `models/` directory

**Problem:** Invalid JSON  
**Solution:** Ensure proper JSON formatting and Content-Type header

**Problem:** Slow response  
**Solution:** Check model file size and server resources

---

**Happy Testing! 🧪**
