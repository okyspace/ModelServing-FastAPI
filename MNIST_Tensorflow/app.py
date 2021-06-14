# Library imports
import uvicorn
from fastapi import FastAPI
from model import predict

'''
Configuration for model microservice
'''
HOST_IP = '127.0.0.1'
HOST_PORT = '8000'


'''
Image supposed to come from client via http post;
Hardcoded data path to simplify the code
'''
DATA = 'data/7.png'


'''
Access auto docs generated: http://127.0.0.1:8000/docs
Run ASGI server: uvicorn app:app --reload
'''

# Create the app object and load model
app = FastAPI()


# Expose predict function
@app.post('/predict')
async def predict_digit():
    '''
    Return json of best predicted class
    '''
    predict_class = predict(DATA)
    return {'predicted': predict_class.item()}


# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host=HOST_IP, port=HOST_PORT)
