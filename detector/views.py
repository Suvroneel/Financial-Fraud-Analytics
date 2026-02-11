"""
Views for fraud detection app
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_utils import FraudDetector


def home(request):
    """
    Home page with prediction form
    """
    context = {
        'title': 'Fraud Detection System',
    }
    return render(request, 'home.html', context)


def predict(request):
    """
    Handle prediction requests
    """
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'amount': float(request.POST.get('amount')),
                'oldbalanceOrg': float(request.POST.get('oldbalanceOrg')),
                'newbalanceOrig': float(request.POST.get('newbalanceOrig')),
                'oldbalanceDest': float(request.POST.get('oldbalanceDest')),
                'newbalanceDest': float(request.POST.get('newbalanceDest')),
                'type': request.POST.get('type'),
            }
            
            # Load model and predict
            detector = FraudDetector()
            result = detector.predict(data)
            
            # Render result page
            context = {
                'title': 'Prediction Result',
                'result': result,
                'transaction': data,
            }
            return render(request, 'result.html', context)
            
        except Exception as e:
            context = {
                'title': 'Error',
                'error': str(e)
            }
            return render(request, 'error.html', context)
    
    return render(request, 'home.html')


@csrf_exempt
def api_predict(request):
    """
    API endpoint for predictions (JSON)
    """
    if request.method == 'POST':
        try:
            # Parse JSON data
            data = json.loads(request.body)
            
            # Validate required fields
            required_fields = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest', 'type']
            
            for field in required_fields:
                if field not in data:
                    return JsonResponse({
                        'error': f'Missing required field: {field}'
                    }, status=400)
            
            # Convert to float
            transaction_data = {
                'amount': float(data['amount']),
                'oldbalanceOrg': float(data['oldbalanceOrg']),
                'newbalanceOrig': float(data['newbalanceOrig']),
                'oldbalanceDest': float(data['oldbalanceDest']),
                'newbalanceDest': float(data['newbalanceDest']),
                'type': data['type'],
            }
            
            # Make prediction
            detector = FraudDetector()
            result = detector.predict(transaction_data)
            
            return JsonResponse({
                'success': True,
                'prediction': result
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON format'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'error': 'Only POST requests allowed'
    }, status=405)


def model_info(request):
    """
    Display model information
    """
    try:
        detector = FraudDetector()
        info = detector.get_model_info()
        
        context = {
            'title': 'Model Information',
            'info': info
        }
        return render(request, 'model_info.html', context)
    except Exception as e:
        context = {
            'title': 'Error',
            'error': str(e)
        }
        return render(request, 'error.html', context)
