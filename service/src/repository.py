import datetime
import os

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Session, select

DATABASE_URL = os.environ["DATABASE_URL"]
Base = declarative_base()

class Duplicate(Base):
    __tablename__ = 'Duplicate'
    uuid = Column(String, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.now)
    link = Column(String, unique=False)
    is_duplicate = Column(Boolean, unique=False, default=False)
    duplicate_for = Column(String, unique=False)
    is_hard = Column(Boolean, unique=False, default=False)
    is_download = Column(Boolean, unique=False, default=False)
    saved = Column(DateTime, default=datetime.datetime.now)


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

def add(item):
    with (Session(engine) as session):
        duplicate = Duplicate(uuid=item[1], created=item[0], link=item[2])
        session.add(duplicate)
        session.commit()
        session.refresh(duplicate)
        id = duplicate.uuid
    return id

def get_duplicate_list_for_download():
    with Session(engine) as session:
        statement = select(Duplicate).where(Duplicate.is_download == False).order_by(Duplicate.uuid).limit(10)
        questions = session.exec(statement)
        result = questions.all()
    return result

def get_all_train():
    with Session(engine) as session:
        statement = select(Duplicate).order_by(Duplicate.saved)
        questions = session.exec(statement)
        result = questions.all()
    return result

def mark_download(id):
    with Session(engine) as session:
        statement = select(Duplicate).where(Duplicate.uuid == id)
        results = session.exec(statement)
        d = results.one()
        d.is_download = True
        session.add(d)
        session.commit()
        session.refresh(d)

def mark_duplicate(id, is_duplicate, duplicate_for):
    with Session(engine) as session:
        statement = select(Duplicate).where(Duplicate.uuid == id)
        results = session.exec(statement)
        d = results.one()
        d.is_duplicate = is_duplicate
        d.duplicate_for = duplicate_for
        session.add(d)
        session.commit()
        session.refresh(d)

def mark_hard(id, is_hard):
    with Session(engine) as session:
        statement = select(Duplicate).where(Duplicate.uuid == id)
        results = session.exec(statement)
        d = results.one()
        d.is_hard = is_hard
        session.add(d)
        session.commit()
        session.refresh(d)

def get_by_id(uuid):
    with Session(engine) as session:
        statement = select(Duplicate).where(Duplicate.uuid == uuid)
        results = session.exec(statement)
        result = results.one()
    return result