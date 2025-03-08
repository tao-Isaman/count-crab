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
    image_data = file.read()
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
        # if "name" in food_info and "นี่ไม่ใช่รูปภาพอาหาร" in food_info["name"]:
        #     response_text = food_info["name"]
        # else:
        #     # Format the nutrition information
        #     food_name = food_info.get("name", "ไม่สามารถระบุชื่ออาหารได้")
        #     protein = food_info.get("protein", "N/A")
        #     carb = food_info.get("carbohydrate", "N/A")
        #     fat = food_info.get("fat", "N/A")
        #     sodium = food_info.get("sodium", "N/A")
        #     calories = food_info.get("calories", "N/A")
            
        #     response_text = (
        #         f"อาหารนี้คือ: {food_name}\n"
        #         f"คุณค่าทางโภชนาการโดยประมาณ:\n"
        #         f"โปรตีน: {protein} กรัม\n"
        #         f"คาร์โบไฮเดรต: {carb} กรัม\n"
        #         f"ไขมัน: {fat} กรัม\n"
        #         f"โซเดียม: {sodium} มิลลิกรัม\n"
        #         f"แคลอรี่: {calories} กิโลแคลอรี่"
        #     )
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=food_info)
        )
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"ขออภัย ไม่สามารถวิเคราะห์ภาพได้: {str(e)}")
        )

from typing import Dict, Any, Optional
from fastapi import Form, Body

@app.post("/debug/send-nutrition")
async def debug_send_nutrition(
    user_id: str = Form(...),
    food_name: str = Form(...),
    protein: Optional[float] = Form(None),
    carbohydrate: Optional[float] = Form(None),
    fat: Optional[float] = Form(None),
    sodium: Optional[float] = Form(None),
    calories: Optional[float] = Form(None)
):
    """
    Debug endpoint to directly send nutritional data to a Line user
    without processing an image.
    
    This helps test the Line messaging API integration and formatting.
    """
    # Create a nutrition info dictionary similar to what classify_with_openai would return
    nutrition_data = {
        "name": food_name,
        "protein": protein if protein is not None else "N/A",
        "carbohydrate": carbohydrate if carbohydrate is not None else "N/A",
        "fat": fat if fat is not None else "N/A",
        "sodium": sodium if sodium is not None else "N/A",
        "calories": calories if calories is not None else "N/A"
    }
    
    try:
        # Format the nutrition information
        response_text = (
            f"อาหารนี้คือ: {nutrition_data['name']}\n"
            f"คุณค่าทางโภชนาการโดยประมาณ:\n"
            f"โปรตีน: {nutrition_data['protein']} กรัม\n"
            f"คาร์โบไฮเดรต: {nutrition_data['carbohydrate']} กรัม\n"
            f"ไขมัน: {nutrition_data['fat']} กรัม\n"
            f"โซเดียม: {nutrition_data['sodium']} มิลลิกรัม\n"
            f"แคลอรี่: {nutrition_data['calories']} กิโลแคลอรี่"
        )
        
        # Send a message to the specified user
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=response_text)
        )
        
        return {"status": "success", "message": "Nutrition data sent to Line user", "data": nutrition_data}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error sending message to Line: {str(e)}"
        )

# Add a JSON body version of the endpoint for easier API testing
@app.post("/debug/send-nutrition-json")
async def debug_send_nutrition_json(
    data: Dict[str, Any] = Body(...)
):
    """
    Debug endpoint accepting JSON body to directly send nutritional data 
    to a Line user without processing an image.
    
    Example JSON body:
    {
        "user_id": "U1234567890abcdef",
        "food_name": "ผัดไทย",
        "protein": 24,
        "carbohydrate": 30,
        "fat": 20,
        "sodium": 10,
        "calories": 20
    }
    """
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Create a nutrition info dictionary
    nutrition_data = {
        "name": data.get("food_name", "ไม่ระบุชื่ออาหาร"),
        "protein": data.get("protein", "N/A"),
        "carbohydrate": data.get("carbohydrate", "N/A"),
        "fat": data.get("fat", "N/A"),
        "sodium": data.get("sodium", "N/A"),
        "calories": data.get("calories", "N/A")
    }
    
    try:
        # Format the nutrition information
        response_text = (
            f"อาหารนี้คือ: {nutrition_data['name']}\n"
            f"คุณค่าทางโภชนาการโดยประมาณ:\n"
            f"โปรตีน: {nutrition_data['protein']} กรัม\n"
            f"คาร์โบไฮเดรต: {nutrition_data['carbohydrate']} กรัม\n"
            f"ไขมัน: {nutrition_data['fat']} กรัม\n"
            f"โซเดียม: {nutrition_data['sodium']} มิลลิกรัม\n"
            f"แคลอรี่: {nutrition_data['calories']} กิโลแคลอรี่"
        )
        
        # Send a message to the specified user
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=response_text)
        )
        
        return {"status": "success", "message": "Nutrition data sent to Line user", "data": nutrition_data}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error sending message to Line: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)