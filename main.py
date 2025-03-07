import os
import requests
import base64
import json

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

# LINE configuration
access_token = os.environ.get("ACCESS_TOKEN")
secret_channel = os.environ.get("SECRET_CHANNEL")

line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret_channel)

# OpenAI configuration
openai_api_key = os.environ.get("OPENAI_API_KEY")

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

async def classify_with_openai(image_data):
    """Use OpenAI's API to classify food images"""
    # Convert binary image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # OpenAI API endpoint
    url = "https://api.openai.com/v1/chat/completions"
    
    # Prepare the payload
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your are image classification machine. i will give you Thai food image you will anser the name of food \nif is not a not a food it will tell \"นี่ไม่ใช่รูปภาพอาหารค่ะ\"\nand show estimation of nuturity (protien, carb, fat, sodium, calories) as json format\n\n## JSON\n{\n\"name\" : \"ผัดไทย\",\n\"protein\" : 24,\n\"carbohydrate\": 30,\n\"fat\": 20,\n\"sodium\": 10,\n\"calories\": 20\n}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "response_format": {
            "type": "text"
        },
        "temperature": 1,
        "max_completion_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    # Send request to OpenAI
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Try to parse JSON from the response content
        try:
            # Check if the response contains JSON (it might be mixed with text)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                food_info = json.loads(json_str)
                return food_info
            else:
                # If no JSON format detected, return the text response
                return {"name": content, "message": "No structured nutrition data available"}
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return {"name": content, "message": "Could not parse nutrition data"}
    else:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Error from OpenAI API: {response.text}"
        )

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    # Process image with OpenAI
    image_data = await file.read()
    try:
        classification_result = classify_with_openai(image_data)
        return classification_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # Simple text response if needed
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="ส่งรูปอาหารมาให้ฉันได้เลยค่ะ จะบอกว่าอาหารอะไรและมีคุณค่าทางโภชนาการเท่าไหร่")
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = message_content.content
    
    try:
        # Use OpenAI to classify the image
        food_info = classify_with_openai(image_data)
        
        # Check if it's food or not
        if "name" in food_info and "นี่ไม่ใช่รูปภาพอาหาร" in food_info["name"]:
            response_text = food_info["name"]
        else:
            # Format the nutrition information
            food_name = food_info.get("name", "ไม่สามารถระบุชื่ออาหารได้")
            protein = food_info.get("protein", "N/A")
            carb = food_info.get("carbohydrate", "N/A")
            fat = food_info.get("fat", "N/A")
            sodium = food_info.get("sodium", "N/A")
            calories = food_info.get("calories", "N/A")
            
            response_text = (
                f"อาหารนี้คือ: {food_name}\n"
                f"คุณค่าทางโภชนาการโดยประมาณ:\n"
                f"โปรตีน: {protein} กรัม\n"
                f"คาร์โบไฮเดรต: {carb} กรัม\n"
                f"ไขมัน: {fat} กรัม\n"
                f"โซเดียม: {sodium} มิลลิกรัม\n"
                f"แคลอรี่: {calories} กิโลแคลอรี่"
            )
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_text)
        )
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"ขออภัย ไม่สามารถวิเคราะห์ภาพได้: {str(e)}")
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)