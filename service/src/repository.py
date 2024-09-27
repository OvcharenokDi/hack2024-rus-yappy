import datetime
import os

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Session

DATABASE_URL = os.environ["DATABASE_URL"]
Base = declarative_base()

class Duplicate(Base):
    __tablename__ = 'Duplicate'
    uuid = Column(String, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.now)
    link = Column(String, unique=False)
    is_duplicate = Column(Boolean, unique=False)
    duplicate_for = Column(String, unique=False)
    is_hard = Column(Boolean, unique=False)


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(engine)
#
def create(item):
    with (Session(engine) as session):
        duplicate = Duplicate(uuid=item[1], created=item[0], link=item[2], is_duplicate=eval(item[3]), duplicate_for=item[4], is_hard=eval(item[5]))
        session.add(duplicate)
        session.commit()
        session.refresh(duplicate)
        id = duplicate.uuid
    return id