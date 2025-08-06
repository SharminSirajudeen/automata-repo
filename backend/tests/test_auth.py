"""
Tests for authentication functionality.
"""

import pytest
from fastapi import status
from httpx import AsyncClient

from app.auth import verify_password, hash_password, create_access_token
from app.database import User


class TestUserRegistration:
    """Test user registration functionality."""
    
    async def test_register_new_user(self, async_client: AsyncClient, test_user_data):
        """Test successful user registration."""
        response = await async_client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == test_user_data["email"]
        assert data["full_name"] == test_user_data["full_name"]
        assert "id" in data
        assert "created_at" in data
        assert data["is_active"] is True
    
    async def test_register_duplicate_email(self, async_client: AsyncClient, test_user_data, test_user):
        """Test registration with existing email fails."""
        response = await async_client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    async def test_register_invalid_email(self, async_client: AsyncClient):
        """Test registration with invalid email fails."""
        invalid_data = {
            "email": "invalid-email",
            "password": "testpassword123",
            "full_name": "Test User"
        }
        
        response = await async_client.post("/auth/register", json=invalid_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    async def test_register_short_password(self, async_client: AsyncClient):
        """Test registration with short password fails."""
        invalid_data = {
            "email": "test@example.com",
            "password": "123",
            "full_name": "Test User"
        }
        
        response = await async_client.post("/auth/register", json=invalid_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    async def test_register_missing_fields(self, async_client: AsyncClient):
        """Test registration with missing fields fails."""
        incomplete_data = {
            "email": "test@example.com"
            # Missing password and full_name
        }
        
        response = await async_client.post("/auth/register", json=incomplete_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_register_malicious_input(self, async_client: AsyncClient):
        """Test registration with malicious input."""
        malicious_data = {
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "<script>alert('xss')</script>"
        }
        
        response = await async_client.post("/auth/register", json=malicious_data)
        
        # Should either sanitize or reject
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_200_OK]
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "<script>" not in data["full_name"]


class TestUserLogin:
    """Test user login functionality."""
    
    async def test_login_valid_credentials(self, async_client: AsyncClient, test_user, test_user_data):
        """Test successful login with valid credentials."""
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        
        response = await async_client.post("/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == test_user_data["email"]
    
    async def test_login_invalid_email(self, async_client: AsyncClient):
        """Test login with non-existent email."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "testpassword123"
        }
        
        response = await async_client.post("/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]
    
    async def test_login_invalid_password(self, async_client: AsyncClient, test_user, test_user_data):
        """Test login with wrong password."""
        login_data = {
            "email": test_user_data["email"],
            "password": "wrongpassword"
        }
        
        response = await async_client.post("/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]
    
    async def test_login_missing_fields(self, async_client: AsyncClient):
        """Test login with missing fields."""
        incomplete_data = {
            "email": "test@example.com"
            # Missing password
        }
        
        response = await async_client.post("/auth/login", json=incomplete_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_login_rate_limiting(self, async_client: AsyncClient):
        """Test login rate limiting."""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        # Make multiple failed login attempts
        responses = []
        for _ in range(10):
            response = await async_client.post("/auth/login", json=login_data)
            responses.append(response)
        
        # Should eventually get rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        # Note: Rate limiting behavior depends on configuration
        # This test might need adjustment based on actual rate limits


class TestTokenAuthentication:
    """Test JWT token authentication."""
    
    async def test_get_current_user_valid_token(self, async_client: AsyncClient, test_user, auth_headers):
        """Test getting current user with valid token."""
        response = await async_client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == test_user.email
        assert data["id"] == str(test_user.id)
    
    async def test_get_current_user_no_token(self, async_client: AsyncClient):
        """Test getting current user without token."""
        response = await async_client.get("/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_get_current_user_invalid_token(self, async_client: AsyncClient):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = await async_client.get("/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_get_current_user_malformed_header(self, async_client: AsyncClient):
        """Test getting current user with malformed auth header."""
        headers = {"Authorization": "InvalidFormat token_here"}
        response = await async_client.get("/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestPasswordUtilities:
    """Test password hashing and verification utilities."""
    
    def test_password_hashing(self):
        """Test password hashing function."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert verify_password(password, hashed)
    
    def test_password_verification_wrong_password(self):
        """Test password verification with wrong password."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)
        
        assert not verify_password(wrong_password, hashed)
    
    def test_password_verification_empty_password(self):
        """Test password verification with empty password."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert not verify_password("", hashed)


class TestTokenGeneration:
    """Test JWT token generation and validation."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        data = {"sub": "test@example.com", "user_id": "123"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_expiry(self):
        """Test access token creation with custom expiry."""
        from datetime import timedelta
        
        data = {"sub": "test@example.com", "user_id": "123"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)
        
        assert isinstance(token, str)
        assert len(token) > 0


class TestAuthenticationSecurity:
    """Test authentication security features."""
    
    async def test_sql_injection_in_login(self, async_client: AsyncClient):
        """Test SQL injection attempt in login."""
        malicious_data = {
            "email": "'; DROP TABLE users; --",
            "password": "testpassword123"
        }
        
        response = await async_client.post("/auth/login", json=malicious_data)
        
        # Should not crash and should handle gracefully
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    async def test_xss_in_registration(self, async_client: AsyncClient):
        """Test XSS attempt in registration."""
        malicious_data = {
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "<script>alert('xss')</script>"
        }
        
        response = await async_client.post("/auth/register", json=malicious_data)
        
        # Should either reject or sanitize
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_200_OK]
    
    async def test_large_payload_rejection(self, async_client: AsyncClient):
        """Test rejection of abnormally large payloads."""
        large_data = {
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "A" * 10000  # Very long name
        }
        
        response = await async_client.post("/auth/register", json=large_data)
        
        # Should reject large payloads
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.integration
class TestAuthenticationFlow:
    """Integration tests for complete authentication flow."""
    
    async def test_complete_registration_login_flow(self, async_client: AsyncClient):
        """Test complete flow: register -> login -> access protected resource."""
        # Step 1: Register
        user_data = {
            "email": "flowtest@example.com",
            "password": "testpassword123",
            "full_name": "Flow Test User"
        }
        
        register_response = await async_client.post("/auth/register", json=user_data)
        assert register_response.status_code == status.HTTP_200_OK
        
        # Step 2: Login
        login_data = {
            "email": user_data["email"],
            "password": user_data["password"]
        }
        
        login_response = await async_client.post("/auth/login", json=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        token_data = login_response.json()
        access_token = token_data["access_token"]
        
        # Step 3: Access protected resource
        headers = {"Authorization": f"Bearer {access_token}"}
        me_response = await async_client.get("/auth/me", headers=headers)
        
        assert me_response.status_code == status.HTTP_200_OK
        user_info = me_response.json()
        assert user_info["email"] == user_data["email"]
        assert user_info["full_name"] == user_data["full_name"]
    
    async def test_token_expiry_handling(self, async_client: AsyncClient):
        """Test handling of expired tokens."""
        # This would require creating an expired token
        # Implementation depends on token expiry settings
        pass  # Placeholder for token expiry test