from fastapi import Request, Depends
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from models import User
from schemas import UserInDB, TokenData
from datetime import datetime, timedelta
from database import get_env_variable, get_db
from typing import Optional
from jose import jwt
import redis
import os
import logging

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = get_env_variable("SECRET_KEY")
ALGORITHM = get_env_variable('ALGORITHM')
redis_client = redis.StrictRedis(host=get_env_variable('REDIS'), port=get_env_variable('REDIS_PORT'), db=0, decode_responses=True)

# Password utilities
def validate_password_policy(password: str):
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password) -> str:
    validate_password_policy(password)
    return pwd_context.hash(password)

# User authentication functions
def get_user(db_session: Session, username: str) -> Optional[User]:
    return db_session.query(User).filter(User.username == username).first()

def authenticate_user(db_session: Session, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db_session, username)
    if not user or not verify_password(password, user.hashed_password):
        logger.info(f"Failed login attempt for username: {username}")
        return None
    logger.info(f"Successful login for username: {username}")
    return UserInDB.from_orm(user)

# 세션 관리
def store_user_session(session_id: str, user: UserInDB):
    user_data = {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email or "",
        #"hashed_password": user.hashed_password or ""
    }
    redis_client.hset(session_id, mapping=user_data)
    redis_client.expire(session_id, 3600)

# JWT creation
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=60))
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Access token issued for {data['sub']} - Expires at {expire}")
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Refresh token issued for {data['sub']} - Expires at {expire}")
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    if token is None:
        raise credentials_exception
    try:
        payload = jwt.decode(token, get_env_variable("SECRET_KEY"), algorithms=[get_env_variable("ALGORITHM")])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data
