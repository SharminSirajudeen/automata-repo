"""
Configuration management for the automata-repo application.
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_title: str = "Theory of Computation Tutor"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    
    # Database Configuration
    database_url: str = "postgresql://automata:password@localhost/automata_db"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_generator_model: str = "codellama:34b"
    ollama_explainer_model: str = "deepseek-coder:33b"
    ollama_vision_model: str = "llava:34b"
    ollama_default_model: str = "llama3.1:8b"
    ollama_timeout: int = 30
    
    # Security Configuration (simplified - no auth needed)
    # secret_key: str = "not-needed-without-auth"
    # access_token_expire_minutes: int = 30
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create a global settings instance
settings = get_settings()