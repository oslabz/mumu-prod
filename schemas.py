from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel

# User Schemas
class UserBase(BaseModel):
    name: str
    username: str
    email: str
    gender: str
    birth: date
    tel: Optional[str] = None
    join_date: Optional[datetime] = None

class UserCreate(UserBase):
    hashed_password: str

class UserInDB(UserBase):
    user_id: int

    class Config:
        from_attributes=True

class User(UserInDB):
    chat_history: List["History"] = []

# Chatroom Schemas
class ChatroomBase(BaseModel):
    title: str
    latest_time: Optional[datetime] = None
    embed_files: Optional[str] = None

class ChatroomCreate(ChatroomBase):
    pass

class ChatroomInDB(ChatroomBase):
    chatroom_id: int

    class Config:
        from_attributes=True

class Chatroom(ChatroomInDB):
    history: List["History"] = []

# History Schemas
class HistoryBase(BaseModel):
    inputs: str
    outputs: str
    timestamp: Optional[datetime] = None
    type: Optional[str] = None
    model: Optional[str] = None

class HistoryCreate(HistoryBase):
    user_id: int
    chatroom_id: int

class HistoryInDB(HistoryBase):
    history_id: int
    user_id: int
    chatroom_id: int

    class Config:
        from_attributes=True

class History(HistoryInDB):
    user: User
    chatroom: Chatroom

# Query Model
class QueryModel(BaseModel):
    inputs: str

# Token Data
class TokenData(BaseModel):
    username: str