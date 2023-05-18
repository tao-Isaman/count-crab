import os
from fastapi import FastAPI, HTTPException, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

app = FastAPI()

access_token = os.environ.get("ACCESS_TOKEN")
secret_channel = os.environ.get("SECRET_CHANNEL")

line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret_channel)

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

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    # Use your function to classify the image
    food_name, carb_estimation = classify_image(message_content.content)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"The food is {food_name} with an estimated {carb_estimation}g of carbs"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Read the PORT environment variable or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
