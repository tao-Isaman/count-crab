import os
import requests

from fastapi import FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

access_token = os.environ.get("ACCESS_TOKEN")
secret_channel = os.environ.get("SECRET_CHANNEL")

line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret_channel)

# crab data 
crab_food = {
        'ข้าวขาหมู':3.3,
        'ข้าวคลุกกะปิ':2.7,
        'ข้าวซอย':2.7,
        'ข้าวผัด':3.1,
        'ข้าวมันไก่':2.3,
        'ข้าวหมกไก่': 3.6,
        'แกงเขียวหวาน': None,
        'แกงเทโพ': None,
        'แกงเลียง': None,
        'แกงจืดเต้าหู้หมูสับ': None,
        'แกงจืดมะระยัดไส้': None,
        'แกงมัสมั่นไก่': None,
        'แกงส้มกุ้ง': None,
        'ไก่ผัดเม็ดมะม่วงหิมพานต์': None,
        'ไข่เจียว': None,
        'ไข่ดาว': None,
        'ไข่พะโล้': None,
    }

@app.post("/webhook")
async def webhook(request: Request):
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()

    # handle webhook body
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature. Please check your channel access token/channel secret.")
        
    return 'OK'



def calculate_insulin(weight: int, carb_portion: float, current_sugar: int):
    CRAB_FACTOR = 15
    EXPECTED_SUGAR = 140 # mg/dL

    icr = 300 / 0.5 * weight
    insulin_senitivity = 1800 * 0.5 / weight

    insulin = (current_sugar - EXPECTED_SUGAR) + ((carb_portion * CRAB_FACTOR) / icr)

    return insulin


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    current_sugar: float = Form(...),
    weight: float = Form(...)
    ):

    
    # Azure endpoint and headers
    endpoint = os.environ.get("AZURE_PREDICT_URL")
    headers = {
        'Prediction-Key': os.environ.get("AZURE_PREDICT_KEY"),
        'Content-Type': 'application/octet-stream'
    }

    # Send image content to Azure Custom Vision
    response = requests.post(endpoint, headers=headers, data=await file.read())

    # Process the response from Azure Custom Vision
    if response.status_code == 200:
        prediction_result = response.json()
        best_prediction = max(prediction_result['predictions'], key=lambda x: x['probability'])
        food_name = best_prediction['tagName']
        carb_estimation = '200'  # Adjust this based on how you get the carb estimation

        carb_partion = crab_food[food_name]
        if carb_partion == None:
            carb_partion = 0


        return {
            'food_name': food_name,
            'carb_estimation': carb_partion * 15,
            'insulin' : calculate_insulin(weight, carb_partion, current_sugar)
        }
    else:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {response.status_code}, {response.text}")


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # line_bot_api.reply_message(
    #     event.reply_token,
    #     TextSendMessage(text=event.message.text))
    pass

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    # Azure endpoint and headers
    endpoint = os.environ.get("AZURE_PREDICT_URL")
    headers = {
        'Prediction-Key': os.environ.get("AZURE_PREDICT_KEY"),
        'Content-Type': 'application/octet-stream'
    }

    # Send image content to Azure Custom Vision
    response = requests.post(endpoint, headers=headers, data=message_content.content)

    # Process the response from Azure Custom Vision
    if response.status_code == 200:
        prediction_result = response.json()
        # Extract information from the result
        # Here I assume the prediction_result has a 'predictions' key with a list of predictions
        # You may need to adjust this based on the actual structure of prediction_result
        best_prediction = max(prediction_result['predictions'], key=lambda x: x['probability'])
        food_name = best_prediction['tagName']
        # carb_estimation = best_prediction['probability']  # Adjust this based on how you get the carb estimation
        carb_estimation = crab_food[food_name] 
        if carb_estimation == None:
            line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"เมนูนี้คือ {food_name} และเมนูนี้ยังไม่มี คาร์โบไฮเดรทในระบบค่ะ"))
        else:
            line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"เมนูนี้คือ {food_name} มีคาร์โบไฮเดรทอยู่ที่ประมาณ {carb_estimation * 15} กรัมค่ะ"))
    else:
        print(f"Error making prediction: {response.status_code}, {response.text}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Read the PORT environment variable or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
