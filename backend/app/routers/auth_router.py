"""
Authentication router for the Automata Learning Platform.
Handles user registration, login, and authentication.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from sqlalchemy.orm import Session
from datetime import timedelta
from ..auth import (
    create_user, authenticate_user, create_access_token,
    get_current_active_user, UserCreate, UserLogin, UserResponse
)
from ..database import get_db, User
from ..config import settings
from ..security import rate_limit_auth, security_logger, validate_input, get_client_info
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse)
@rate_limit_auth("register")
async def register(
    request: Request,
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    try:
        client_info = get_client_info(request)
        
        # Validate and sanitize inputs
        validate_input(user_create.email, "email")
        validate_input(user_create.full_name, "safe_string", max_length=100)
        
        user = create_user(db, user_create)
        
        # Log successful registration
        security_logger.log_auth_attempt(
            user.email, True, client_info["ip"], client_info["user_agent"]
        )
        
        logger.info(f"User registered successfully: {user.email}")
        return UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            skill_level=user.skill_level,
            created_at=user.created_at
        )
    except Exception as e:
        client_info = get_client_info(request)
        security_logger.log_auth_attempt(
            user_create.email if hasattr(user_create, 'email') else "unknown", 
            False, client_info["ip"], client_info["user_agent"]
        )
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed"
        )


@router.post("/login")
@rate_limit_auth("login")
async def login(
    request: Request,
    user_login: UserLogin,
    db: Session = Depends(get_db)
):
    """Login and receive access token."""
    client_info = get_client_info(request)
    
    # Validate input
    validate_input(user_login.email, "email")
    
    user = authenticate_user(db, user_login.email, user_login.password)
    if not user:
        # Log failed login attempt
        security_logger.log_auth_attempt(
            user_login.email, False, client_info["ip"], client_info["user_agent"]
        )
        logger.warning(f"Failed login attempt for email: {user_login.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires
    )
    
    # Log successful login
    security_logger.log_auth_attempt(
        user.email, True, client_info["ip"], client_info["user_agent"]
    )
    
    logger.info(f"User logged in successfully: {user.email}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            skill_level=user.skill_level,
            created_at=user.created_at
        )
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        skill_level=current_user.skill_level,
        created_at=current_user.created_at
    )