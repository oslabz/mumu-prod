import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Get environment variables
def get_env_variable(var_name: str, default: str = None) -> str:
    value = os.getenv(var_name, default)
    if not value:
        logger.error(f"{var_name} environment variable not set or incorrect.")
    else:
        logger.info(f"{var_name} Success")
    return value

DATABASE_URL = f"mysql+pymysql://{get_env_variable('DB_USER')}:{get_env_variable('DB_PASSWORD')}@{get_env_variable('DB_HOST')}/{get_env_variable('DB_NAME')}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
