from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from datetime import datetime
from models import User, History, Chatroom
from schemas import UserInDB, TokenData, UserBase
from utils import authenticate_user, create_access_token, create_refresh_token, get_password_hash, validate_password_policy, store_user_session
from database import get_db, get_env_variable
from jose import jwt, JWTError
import logging
import json
import time
import redis # type: ignore
import os
from pydantic import ValidationError

router = APIRouter(prefix="/login", tags=["login"])
logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
redis_client = redis.StrictRedis(host=get_env_variable('REDIS'), port=get_env_variable('REDIS_PORT'), db=0, decode_responses=True)

ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour expiration

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, get_env_variable("SECRET_KEY"), algorithms=[get_env_variable("ALGORITHM")])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data

def get_current_user(request: Request, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserInDB:
    # JWT 토큰 검증
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(token, credentials_exception)
    
    # 블랙리스트 확인
    if redis_client.get(f"logoutlist_{token}"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is logout")

    # 데이터베이스에서 사용자 정보 조회
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception

    # 쿠키 기반 세션 확인
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_data = redis_client.hgetall(session_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")
    logger.debug(f"Session ID: {session_id}")
    logger.debug(f"User Data from Redis: {user_data}")
    
    try:
        user_data_converted = {
            "id": int(user_data.get("id", 0)),
            "username": user_data.get("username", ""),
            "email": user_data.get("email", ""),
            "full_name": user_data.get("full_name", ""),
            "disabled": user_data.get("disabled") == "True",
            "hashed_password": user_data.get("hashed_password", "")
        }
        if not user_data_converted["hashed_password"]:
            logger.error("hashed_password is missing")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user data in session")
        return UserInDB(**user_data_converted)
    except (ValueError, TypeError) as e:
        logger.error(f"Conversion Error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user data in session")
    except ValidationError as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user data in session")

def validate_gender(gender: str):
    allowed_genders = ['Male', 'Female', 'Other']
    if gender not in allowed_genders:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid gender value: {gender}")

@router.post("/sign_in")
async def sign_in_user(name: str, username: str, email: str, password: str, gender: str, birth: datetime, tel: str = None, db: Session = Depends(get_db)):
    try:
        validate_gender(gender)
        validate_password_policy(password)
        hashed_password = get_password_hash(password)
        new_user = User(
            name=name,
            username=username,
            email=email,
            hashed_password=hashed_password,
            gender=gender,
            birth=birth,
            tel=tel
        )
        db.add(new_user)
        db.commit()
        logger.info(f"New user registered: {username}")
        return {"Username": username, "email": email}
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    session_id = os.urandom(24).hex()
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True, samesite="Lax", max_age=3600)
    store_user_session(session_id, user)
    
    subquery = db.query(History.user_id, func.max(History.timestamp).label('latest_timestamp')).filter(
        History.user_id == user.user_id).group_by(func.date(History.timestamp)).subquery()
    chat_history_query = db.query(History).join(
        subquery, (History.user_id == subquery.c.user_id) & (History.timestamp == subquery.c.latest_timestamp)).order_by(History.timestamp.desc())
    user_chat_history = [history.to_dict() for history in chat_history_query.all()]
    
    redis_client.set(f"{session_id}_chat_history", json.dumps(jsonable_encoder(user_chat_history)))
    redis_client.expire(f"{session_id}_chat_history", time)
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})

    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True, samesite="Lax", max_age=3600)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True, samesite="Lax", max_age=3600)
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=True, samesite="Lax", max_age=604800)  # 7 days

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer", "chat_history": user_chat_history}

@router.post("/logout")
async def logout_user(request: Request, response: Response, db: Session = Depends(get_db)):
    # 요청의 쿠키에서 액세스 토큰과 리프레시 토큰 가져오기
    access_token = request.cookies.get("access_token")
    refresh_token = request.cookies.get("refresh_token")
    if access_token is None or refresh_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No token found")

    logger.debug(f"Received logout request with access token: {access_token} and refresh token: {refresh_token}")

    # 토큰 검증
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(access_token, credentials_exception)

    # Redis에 액세스 토큰과 리프레시 토큰을 블랙리스트에 추가
    redis_client.set(f"logoutlist_{access_token}", "true")
    redis_client.expire(f"logoutlist_{access_token}", ACCESS_TOKEN_EXPIRE_MINUTES * 15)
    redis_client.set(f"logoutlist_{refresh_token}", "true")
    redis_client.expire(f"logoutlist_{refresh_token}", ACCESS_TOKEN_EXPIRE_MINUTES * 168)

    # 세션 ID 가져오기 및 Redis에서 세션 데이터 삭제
    session_id = request.cookies.get("session_id")
    if session_id:
        redis_client.delete(session_id)

    # 세션 쿠키 및 토큰 쿠키 삭제
    response.delete_cookie(key="session_id")
    response.delete_cookie(key="access_token")
    response.delete_cookie(key="refresh_token")

    logger.info(f"Logged out user: {token_data.username}")
    return {"message": "Logged out successfully"}