import os
import sys
import requests
import base64
import json
import logging

from fastapi import (
    FastAPI, 
    HTTPException, 
    Request, 
    File, 
    Form, 
    UploadFile, 
    Body
)
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

# LINE Bot SDK imports
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, 
    TextMessage, 
    TextSendMessage, 
    ImageMessage
)
from linebot.models import FlexSendMessage

# ------------------------------------------------------------------------------
# Configure Logging (structured for Google Cloud, also works locally)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------
app = FastAPI()

# Enable CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Environment Variables / Config
# ------------------------------------------------------------------------------
access_token = os.environ.get("ACCESS_TOKEN")       # LINE channel access token
secret_channel = os.environ.get("SECRET_CHANNEL")   # LINE channel secret
openai_api_key = os.environ.get("OPENAI_API_KEY")   # OpenAI API key

if not access_token or not secret_channel:
    logging.warning("Warning: ACCESS_TOKEN or SECRET_CHANNEL is not set.")

if not openai_api_key:
    logging.warning("Warning: OPENAI_API_KEY is not set.")

line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret_channel)

# ------------------------------------------------------------------------------
# Utility: Structured Logging to Cloud Run (Synchronous)
# ------------------------------------------------------------------------------
def log_food_info(food_info: dict):
    """
    Logs food information in a structured JSON format for Cloud Logging.
    """
    log_data = {
        "severity": "INFO",
        "message": "Food classification result",
        "food_info": food_info
    }
    logging.info(json.dumps(log_data, ensure_ascii=False))

# ------------------------------------------------------------------------------
# OpenAI Image Classification (Synchronous)
# ------------------------------------------------------------------------------
def classify_with_openai(image_data: bytes) -> dict:
    """
    Use OpenAI's API to classify food images (synchronously).
    """
    # Convert binary image to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # OpenAI API endpoint
    url = "https://api.openai.com/v1/chat/completions"

    # Prepare the payload
    payload = {
        "model": "gpt-4o",  # Example model name, adjust to your usage
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an image classification machine. I will give you a Thai food image; "
                            "you will answer the name of the food.\n"
                            'If it is not a food image, respond with "นี่ไม่ใช่รูปภาพอาหารค่ะ".\n'
                            "Also return an estimate of nutrition (protein, carb, fat, sodium, calories, materials, details)"
                            "materials is meal core materials of food. details is mean advice message about food is field are need long message"
                            "in JSON format.\n\n"
                            "## JSON Example\n"
                            "{\n"
                            '"name": "ผัดไทย",\n'
                            '"protein": 24,\n'
                            '"carbohydrate": 30,\n'
                            '"fat": 20,\n'
                            '"sodium": 10,\n'
                            '"calories": 20\n'
                            '"materials": "เส้นหมี่, หมู, ผัก, พริก, ซอสปรุงรส,\n'
                            '"details": "ผัดไทยเป็นอาหารที่มีคุณค่าทางโภชนาการครบถ้วน แต่ควรรับประทานในปริมาณที่เหมาะสม เนื่องจากมีแป้งและน้ำมันค่อนข้างมาก ทานคู่กับผักสดที่เสิร์ฟมาด้วยเพื่อเพิ่มใยอาหารและวิตามิน บีบมะนาวเพิ่มรสชาติแทนการเติมน้ำตาลหรือน้ำปลา เพื่อลดปริมาณโซเดียมและน้ำตาล ผัดไทยเป็นอาหารที่ให้พลังงานค่อนข้างสูง จึงเหมาะสำหรับมื้อกลางวันมากกว่ามื้อเย็น สำหรับผู้ที่ควบคุมน้ำหนัก ควรรับประทานในปริมาณครึ่งจาน และเพิ่มสัดส่วนผักเคียงให้มากขึ้น การทานผัดไทยหลังออกกำลังกายจะช่วยเติมพลังงานได้ดี เนื่องจากมีทั้งโปรตีนจากไข่และกุ้ง และคาร์โบไฮเดรตจากเส้น'
                            "}\n"
                        )
                    }
                ],
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
                ],
            },
        ],
        "response_format": {"type": "text"},
        "temperature": 1,
        "max_completion_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Make synchronous request
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Error from OpenAI API: {response.text}"
        )

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Attempt to parse JSON out of the content
    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            food_info = json.loads(json_str)
            return food_info
        else:
            # If no recognizable JSON found
            return {"name": content, "message": "No structured nutrition data available"}
    except json.JSONDecodeError:
        return {"name": content, "message": "Could not parse nutrition data"}

# ------------------------------------------------------------------------------
# LINE Webhook Endpoint
# ------------------------------------------------------------------------------
@app.post("/webhook")
async def webhook(request: Request):
    """
    LINE Messaging API webhook endpoint.
    """
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid signature. Check your channel access token/channel secret."
        )
    return "OK"

# ------------------------------------------------------------------------------
# /classify Endpoint (Direct API for Testing)
# ------------------------------------------------------------------------------
@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns classification/nutrition info (via OpenAI).
    """
    image_data = file.file.read()
    try:
        classification_result = classify_with_openai(image_data)
        return classification_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")

# ------------------------------------------------------------------------------
# LINE TextMessage Handler
# ------------------------------------------------------------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    """
    Simple text response prompting the user to send an image.
    """
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=(
            "ส่งรูปอาหารมาให้ฉันได้เลยค่ะ "
            "ฉันจะบอกชื่ออาหารและคุณค่าทางโภชนาการให้"
        ))
    )

# ------------------------------------------------------------------------------
# LINE ImageMessage Handler (Synchronous)
# ------------------------------------------------------------------------------
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event: MessageEvent):
    """
    Handles an incoming image message: calls OpenAI to classify, logs info, and responds.
    """
    try:
        # 1) Download the image data
        message_content = line_bot_api.get_message_content(event.message.id)
        image_data = message_content.content

        # 2) Classify the image (synchronously)
        food_info = classify_with_openai(image_data)

        # 3) If `food_info` is still a JSON string, parse it
        if isinstance(food_info, str):
            try:
                food_info = json.loads(food_info)
            except json.JSONDecodeError:
                logging.error("Error: food_info is not a valid JSON string")
                food_info = {"name": "ไม่สามารถระบุชื่ออาหารได้"}

        # 4) Log the info
        log_food_info(food_info)

        # # 5) Build response text
        # food_name = food_info.get("name", "ไม่สามารถระบุชื่ออาหารได้")
        # protein = food_info.get("protein", "N/A")
        # carb = food_info.get("carbohydrate", "N/A")
        # fat = food_info.get("fat", "N/A")
        # sodium = food_info.get("sodium", "N/A")
        # calories = food_info.get("calories", "N/A")

        # response_text = (
        #     f"อาหารนี้คือ: {food_name}\n"
        #     f"คุณค่าทางโภชนาการโดยประมาณ:\n"
        #     f"โปรตีน: {protein} กรัม\n"
        #     f"คาร์โบไฮเดรต: {carb} กรัม\n"
        #     f"ไขมัน: {fat} กรัม\n"
        #     f"โซเดียม: {sodium} มิลลิกรัม\n"
        #     f"แคลอรี่: {calories} กิโลแคลอรี่"
        # )

        flex_message = create_flex_nutrition_message(food_info)

        # 6) Reply to user
        line_bot_api.reply_message(
            event.reply_token,
            flex_message
        )

    except Exception as e:
        error_message = f"ขออภัย ไม่สามารถวิเคราะห์ภาพได้: {str(e)}"
        logging.error(json.dumps({"severity": "ERROR", "message": error_message}))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=error_message)
        )

# ------------------------------------------------------------------------------
# Debug Endpoints to Test Sending Nutrition to a User
# ------------------------------------------------------------------------------
@app.post("/debug/send-nutrition")
def debug_send_nutrition(
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
    """
    nutrition_data = {
        "name": food_name,
        "protein": protein if protein is not None else "N/A",
        "carbohydrate": carbohydrate if carbohydrate is not None else "N/A",
        "fat": fat if fat is not None else "N/A",
        "sodium": sodium if sodium is not None else "N/A",
        "calories": calories if calories is not None else "N/A"
    }
    
    try:
        response_text = (
            f"อาหารนี้คือ: {nutrition_data['name']}\n"
            f"คุณค่าทางโภชนาการโดยประมาณ:\n"
            f"โปรตีน: {nutrition_data['protein']} กรัม\n"
            f"คาร์โบไฮเดรต: {nutrition_data['carbohydrate']} กรัม\n"
            f"ไขมัน: {nutrition_data['fat']} กรัม\n"
            f"โซเดียม: {nutrition_data['sodium']} มิลลิกรัม\n"
            f"แคลอรี่: {nutrition_data['calories']} กิโลแคลอรี่"
        )

        line_bot_api.push_message(user_id, TextSendMessage(text=response_text))
        return {
            "status": "success",
            "message": "Nutrition data sent to Line user",
            "data": nutrition_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error sending message to Line: {str(e)}"
        )

@app.post("/debug/send-nutrition-json")
def debug_send_nutrition_json(data: Dict[str, Any] = Body(...)):
    """
    Debug endpoint accepting JSON to directly send nutritional data to a Line user.
    JSON Example:
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
    
    nutrition_data = {
        "name": data.get("food_name", "ไม่ระบุชื่ออาหาร"),
        "protein": data.get("protein", "N/A"),
        "carbohydrate": data.get("carbohydrate", "N/A"),
        "fat": data.get("fat", "N/A"),
        "sodium": data.get("sodium", "N/A"),
        "calories": data.get("calories", "N/A")
    }
    
    try:
        response_text = (
            f"อาหารนี้คือ: {nutrition_data['name']}\n"
            f"คุณค่าทางโภชนาการโดยประมาณ:\n"
            f"โปรตีน: {nutrition_data['protein']} กรัม\n"
            f"คาร์โบไฮเดรต: {nutrition_data['carbohydrate']} กรัม\n"
            f"ไขมัน: {nutrition_data['fat']} กรัม\n"
            f"โซเดียม: {nutrition_data['sodium']} มิลลิกรัม\n"
            f"แคลอรี่: {nutrition_data['calories']} กิโลแคลอรี่"
        )
        
        line_bot_api.push_message(user_id, TextSendMessage(text=response_text))
        
        return {
            "status": "success",
            "message": "Nutrition data sent to Line user",
            "data": nutrition_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error sending message to Line: {str(e)}"
        )

def create_flex_nutrition_message(food_info):
    """
    Generates a LINE Flex Message for displaying food nutrition details.
    
    :param food_info: Dictionary containing food details.
    :return: FlexSendMessage object
    """

    flex_message = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": food_info.get("name", "ไม่สามารถระบุชื่ออาหารได้"),
                    "weight": "bold",
                    "size": "xl",
                    "margin": "md"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": "แคลอรี่:", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                {"type": "text", "text": f"{food_info.get('calories', 'N/A')} กิโลแคลอรี่", "wrap": True, "color": "#666666", "size": "sm", "flex": 3}
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": "โปรตีน:", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                {"type": "text", "text": f"{food_info.get('protein', 'N/A')} กรัม", "wrap": True, "color": "#666666", "size": "sm", "flex": 3}
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": "ไขมัน:", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                {"type": "text", "text": f"{food_info.get('fat', 'N/A')} กรัม", "wrap": True, "color": "#666666", "size": "sm", "flex": 3}
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": "คาร์โบไฮเดรต:", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                {"type": "text", "text": f"{food_info.get('carbohydrate', 'N/A')} กรัม", "wrap": True, "color": "#666666", "size": "sm", "flex": 3}
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": "โซเดียม:", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                {"type": "text", "text": f"{food_info.get('sodium', 'N/A')} มิลลิกรัม", "wrap": True, "color": "#666666", "size": "sm", "flex": 3}
                            ]
                        }
                    ]
                },
                {
                    "type": "separator",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "วัตถุดิบ",
                    "weight": "bold",
                    "size": "md",
                    "margin": "lg",
                    "color": "#1DB446"
                },
                {
                    "type": "text",
                    "text": f"{food_info.get('materials', 'ข้อมูลวัตถุกำลังอยู่ในระหว่างการพัฒนา')}",
                    "wrap": True,
                    "size": "sm",
                    "color": "#666666",
                    "margin": "sm"
                },
                {
                    "type": "separator",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "คำแนะนำ",
                    "weight": "bold",
                    "size": "md",
                    "margin": "lg",
                    "color": "#1DB446"
                },
                {
                    "type": "text",
                    "text": f"{food_info.get('details', 'ข้อมูลคำแนะนำกำลังอยู่ในระหว่างการพัฒนา')}",
                    "wrap": True,
                    "size": "sm",
                    "color": "#666666",
                    "margin": "sm"
                },
                {
                    "type": "separator",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "Note: ข้อมูลนี้เป็นการประมาณค่าจาก AI และอาจมีความคลาดเคลื่อนได้",
                    "wrap": True,
                    "size": "xs",
                    "color": "#FF6B6E",
                    "margin": "lg"
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#1DB446",
                    "action": {
                        "type": "uri",
                        "label": "แนะนำอาหารที่คล้ายกัน",
                        "uri": "https://example.com/edit"
                    }
                }
            ]
        }
    }

    return FlexSendMessage(alt_text="ข้อมูลโภชนาการ", contents={"type": "carousel", "contents": [flex_message]})



# ------------------------------------------------------------------------------
# Uvicorn Entry Point (if running locally or Docker without Gunicorn)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
