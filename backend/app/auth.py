"""
Authentication module using Supabase.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from .database import get_db, User, get_or_create_user
from .config import settings
import httpx
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
bearer_scheme = HTTPBearer()


class TokenData(BaseModel):
    """Token data model."""
    email: Optional[str] = None
    user_id: Optional[str] = None


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    skill_level: str
    created_at: datetime


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=["HS256"]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        
        token_data = TokenData(email=email, user_id=payload.get("user_id"))
        
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, user_create: UserCreate) -> User:
    """Create a new user."""
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_create.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_create.password)
    user = User(
        email=user_create.email,
        hashed_password=hashed_password,
        full_name=user_create.full_name
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user


# Optional: Supabase integration for future enhancement
class SupabaseAuth:
    """Supabase authentication client."""
    
    def __init__(self, url: str, anon_key: str):
        self.url = url
        self.anon_key = anon_key
        self.client = httpx.AsyncClient()
    
    async def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """Sign up a new user with Supabase."""
        response = await self.client.post(
            f"{self.url}/auth/v1/signup",
            json={"email": email, "password": password},
            headers={
                "apikey": self.anon_key,
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in a user with Supabase."""
        response = await self.client.post(
            f"{self.url}/auth/v1/token?grant_type=password",
            json={"email": email, "password": password},
            headers={
                "apikey": self.anon_key,
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a Supabase token."""
        response = await self.client.get(
            f"{self.url}/auth/v1/user",
            headers={
                "apikey": self.anon_key,
                "Authorization": f"Bearer {token}"
            }
        )
        response.raise_for_status()
        return response.json()