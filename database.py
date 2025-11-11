from sqlalchemy import create_engine, Column, Integer, String, JSON, select
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    _tablename_ = "users"
    id = Column(Integer, primary_key=True)
    provider = Column(String)
    provider_user_id = Column(String, unique=True)
    name = Column(String)
    email = Column(String)
    avatar = Column(String)
    data = Column(JSON)

Base.metadata.create_all(engine)

def get_user_by_provider(provider, provider_user_id):
    session = SessionLocal()
    user = session.execute(
        select(User).where(User.provider == provider, User.provider_user_id == provider_user_id)
    ).scalar_one_or_none()
    session.close()
    return user

def create_user(provider, provider_user_id, name, email, avatar):
    session = SessionLocal()
    new_user = User(
        provider=provider,
        provider_user_id=provider_user_id,
        name=name,
        email=email,
        avatar=avatar,
        data={}
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    session.close()
    return new_user

def save_user_data(user_id, new_data):
    session = SessionLocal()
    user = session.get(User, user_id)
    if user:
        user.data = new_data
        session.commit()
    session.close()

def load_user_data(user_id):
    session = SessionLocal()
    user = session.get(User, user_id)
    data = user.data if user else {}
    session.close()
    return data