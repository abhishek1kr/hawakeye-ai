# BUG-14 FIX: Use modern SQLAlchemy 2.0-compatible DeclarativeBase
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime, timezone
from passlib.context import CryptContext
import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Use DATABASE_URL from .env if available, otherwise fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback to SQLite if DATABASE_URL is not set or USE_SQLITE is 1
if not DATABASE_URL or os.getenv("USE_SQLITE") == "1":
    # Use absolute path to prevent db resetting when running from different directories
    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    DATABASE_URL = f"sqlite:///{base_dir}/hawkeye_ai.db"

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# BUG-14 FIX: Use new-style declarative base class
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(200))
    # BUG-13 FIX: Use timezone-aware datetime
    created_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    
    projects = relationship("Project", back_populates="user")

    def verify_password(self, plain_password):
        return pwd_context.verify(plain_password, self.hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String(100), primary_key=True, index=True)  # job_id
    user_id = Column(Integer, ForeignKey("users.id"))
    video_name = Column(String(255))
    # BUG-13 FIX: Use timezone-aware datetime
    processed_at = Column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    
    # Summary Metrics
    safety_score = Column(Float)
    risk_level = Column(String(20))
    road_width_avg = Column(Float)
    total_estimated_cost = Column(Float)
    
    # Store full report as JSON for easy retrieval
    full_report_json = Column(JSON)
    
    user = relationship("User", back_populates="projects")
    # BUG-05 FIX: Move frames relationship into class body (not monkey-patched after)
    frames = relationship("FrameMetric", order_by="FrameMetric.frame_idx", back_populates="project")

class FrameMetric(Base):
    __tablename__ = "frame_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String(100), ForeignKey("projects.id"))
    frame_idx = Column(Integer)
    timestamp = Column(Float)
    lat = Column(Float)
    lon = Column(Float)
    safety_score = Column(Float)
    road_width = Column(Float)
    
    project = relationship("Project", back_populates="frames")

# BUG-05 FIX: Removed monkey-patch — frames relationship is now declared inside Project class above

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
