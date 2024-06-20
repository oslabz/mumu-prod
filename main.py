from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import answer
import login
#import file
#import pdf
#import redhat
import uvicorn
import logging
import AI

# Initialize FastAPI app
app = FastAPI()

# Include the router
app.include_router(login.router)
#app.include_router(pdf.router)
#app.include_router(redhat.router)
app.include_router(answer.router)
#app.include_router(file.router)
app.include_router(AI.router)

# Add session middleware with a 1-hour session expiry
app.add_middleware(
    SessionMiddleware, 
    secret_key='mumulbo', 
    max_age=36000
)

# CORSMiddleware 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"]  # 모든 헤더 노출
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)



# # Save chat history
# new_chat_history = login.ChatHistory(
#     user_id=current_user.id,
#     inputs=inputs,
#     outputs=output_text,
#     timestamp=datetime.utcnow()
# )
# db.add(new_chat_history)
# db.commit()
