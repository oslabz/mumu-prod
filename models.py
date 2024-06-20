from sqlalchemy import Column, Integer, String, Date, Enum, ForeignKey, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    gender = Column(Enum('Male', 'Female', 'Other'), nullable=False)
    birth = Column(Date, nullable=False)
    tel = Column(String(15))
    join_date = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    
    chat_history = relationship("History", back_populates="user")

class Chatroom(Base):
    __tablename__ = 'chatroom'
    chatroom_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    latest_time = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    embed_files = Column(Text)
    
    history = relationship("History", back_populates="chatroom")

class History(Base):
    __tablename__ = 'history'
    history_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    chatroom_id = Column(Integer, ForeignKey('chatroom.chatroom_id'), nullable=False)
    inputs = Column(Text, nullable=False)
    outputs = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    type = Column(String(50))
    model = Column(String(50))
    
    user = relationship("User", back_populates="chat_history")
    chatroom = relationship("Chatroom", back_populates="history")
