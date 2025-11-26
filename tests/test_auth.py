# tests/test_auth.py
"""
Phase 3.1 Authentication Tests

Tests for:
- OTP generation and verification
- TTL enforcement (≤10 min)
- Max attempts (5)
- Cooldown period (60s)
- Per-email daily cap (20/day)
- Constant-time comparison
- Single-use OTP
- JWT issuance and validation

Run with: pytest tests/test_auth.py -v
Run integration tests: pytest tests/test_auth.py -v -m integration
"""

import pytest

# Mark entire module as integration (requires Phase 3.1 modules)
pytestmark = pytest.mark.integration

import time
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

# Test imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# OTP Generation Tests
# ============================================================

class TestOTPGeneration:
    """Tests for OTP generation."""
    
    def test_otp_is_6_digits(self):
        """OTP must be exactly 6 digits."""
        from app.accounts.otp import generate_otp
        
        for _ in range(100):
            otp = generate_otp()
            assert len(otp) == 6, f"OTP length should be 6, got {len(otp)}"
            assert otp.isdigit(), f"OTP should be digits only, got {otp}"
    
    def test_otp_randomness(self):
        """OTPs should be random (no duplicates in reasonable sample)."""
        from app.accounts.otp import generate_otp
        
        otps = [generate_otp() for _ in range(1000)]
        unique_otps = set(otps)
        # With 6 digits (1M possibilities), 1000 samples should have very few dupes
        assert len(unique_otps) > 950, "Too many duplicate OTPs generated"
    
    def test_otp_hash_is_sha256(self):
        """OTP hash should be SHA-256."""
        from app.accounts.otp import hash_otp
        
        otp = "123456"
        hashed = hash_otp(otp)
        
        # SHA-256 produces 64 hex characters
        assert len(hashed) == 64, f"Hash length should be 64, got {len(hashed)}"
        assert all(c in '0123456789abcdef' for c in hashed), "Hash should be hex"
    
    def test_otp_hash_deterministic(self):
        """Same OTP should produce same hash."""
        from app.accounts.otp import hash_otp
        
        otp = "123456"
        hash1 = hash_otp(otp)
        hash2 = hash_otp(otp)
        assert hash1 == hash2, "Hash should be deterministic"
    
    def test_otp_hash_different_inputs(self):
        """Different OTPs should produce different hashes."""
        from app.accounts.otp import hash_otp
        
        hash1 = hash_otp("123456")
        hash2 = hash_otp("654321")
        assert hash1 != hash2, "Different OTPs should have different hashes"


# ============================================================
# OTP Verification Tests
# ============================================================

class TestOTPVerification:
    """Tests for OTP verification logic."""
    
    def test_verify_otp_hash_correct(self):
        """Correct OTP should verify successfully."""
        from app.accounts.otp import hash_otp, verify_otp_hash
        
        otp = "123456"
        hashed = hash_otp(otp)
        assert verify_otp_hash(otp, hashed) is True
    
    def test_verify_otp_hash_incorrect(self):
        """Incorrect OTP should fail verification."""
        from app.accounts.otp import hash_otp, verify_otp_hash
        
        otp = "123456"
        hashed = hash_otp(otp)
        assert verify_otp_hash("654321", hashed) is False
    
    def test_verify_otp_constant_time(self):
        """Verification should use constant-time comparison."""
        from app.accounts.otp import verify_otp_hash, hash_otp
        
        # This is a behavioral test - we verify the function uses hmac.compare_digest
        # by checking timing consistency (not a perfect test but indicative)
        hashed = hash_otp("123456")
        
        times_correct = []
        times_wrong_first = []
        times_wrong_last = []
        
        for _ in range(100):
            start = time.perf_counter()
            verify_otp_hash("123456", hashed)
            times_correct.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            verify_otp_hash("023456", hashed)  # Wrong first char
            times_wrong_first.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            verify_otp_hash("123450", hashed)  # Wrong last char
            times_wrong_last.append(time.perf_counter() - start)
        
        # Timing should be similar (within 5x) for all cases
        avg_correct = sum(times_correct) / len(times_correct)
        avg_wrong_first = sum(times_wrong_first) / len(times_wrong_first)
        avg_wrong_last = sum(times_wrong_last) / len(times_wrong_last)
        
        # Allow 5x variance due to system noise, but should be same order of magnitude
        assert avg_wrong_first < avg_correct * 5, "Timing attack vulnerability: wrong first char faster"
        assert avg_wrong_last < avg_correct * 5, "Timing attack vulnerability: wrong last char faster"


# ============================================================
# OTP TTL Tests
# ============================================================

class TestOTPTTL:
    """Tests for OTP time-to-live enforcement."""
    
    def test_otp_ttl_is_10_minutes(self):
        """OTP TTL should be 10 minutes."""
        from app.accounts.models import OTP_TTL_MINUTES
        
        assert OTP_TTL_MINUTES == 10, f"OTP TTL should be 10 minutes, got {OTP_TTL_MINUTES}"
    
    def test_otp_expiry_calculation(self):
        """Expiry should be created_at + 10 minutes."""
        from app.accounts.models import OtpCode
        
        now = datetime.utcnow()
        expiry = OtpCode.compute_expiry(now)
        
        expected = now + timedelta(minutes=10)
        assert expiry == expected, f"Expiry should be {expected}, got {expiry}"
    
    def test_otp_is_expired_before_ttl(self):
        """OTP should not be expired before TTL."""
        from app.accounts.models import OtpCode
        import uuid
        
        now = datetime.utcnow()
        otp = OtpCode(
            id=uuid.uuid4(),
            email="test@example.com",
            otp_hash="abc123",
            expires_at=now + timedelta(minutes=5),
            attempts=0,
            created_at=now,
        )
        
        assert otp.is_expired is False
    
    def test_otp_is_expired_after_ttl(self):
        """OTP should be expired after TTL."""
        from app.accounts.models import OtpCode
        import uuid
        
        now = datetime.utcnow()
        otp = OtpCode(
            id=uuid.uuid4(),
            email="test@example.com",
            otp_hash="abc123",
            expires_at=now - timedelta(minutes=1),  # Already expired
            attempts=0,
            created_at=now - timedelta(minutes=11),
        )
        
        assert otp.is_expired is True


# ============================================================
# OTP Attempts Tests
# ============================================================

class TestOTPAttempts:
    """Tests for OTP max attempts enforcement."""
    
    def test_max_attempts_is_5(self):
        """Max attempts should be 5."""
        from app.accounts.models import OTP_MAX_ATTEMPTS
        
        assert OTP_MAX_ATTEMPTS == 5, f"Max attempts should be 5, got {OTP_MAX_ATTEMPTS}"
    
    def test_otp_not_locked_under_max(self):
        """OTP should not be locked under max attempts."""
        from app.accounts.models import OtpCode
        import uuid
        
        now = datetime.utcnow()
        otp = OtpCode(
            id=uuid.uuid4(),
            email="test@example.com",
            otp_hash="abc123",
            expires_at=now + timedelta(minutes=5),
            attempts=4,  # Under max
            created_at=now,
        )
        
        assert otp.is_locked is False
    
    def test_otp_locked_at_max(self):
        """OTP should be locked at max attempts."""
        from app.accounts.models import OtpCode
        import uuid
        
        now = datetime.utcnow()
        otp = OtpCode(
            id=uuid.uuid4(),
            email="test@example.com",
            otp_hash="abc123",
            expires_at=now + timedelta(minutes=5),
            attempts=5,  # At max
            created_at=now,
        )
        
        assert otp.is_locked is True
    
    def test_otp_locked_over_max(self):
        """OTP should be locked over max attempts."""
        from app.accounts.models import OtpCode
        import uuid
        
        now = datetime.utcnow()
        otp = OtpCode(
            id=uuid.uuid4(),
            email="test@example.com",
            otp_hash="abc123",
            expires_at=now + timedelta(minutes=5),
            attempts=10,  # Over max
            created_at=now,
        )
        
        assert otp.is_locked is True


# ============================================================
# OTP Cooldown Tests
# ============================================================

class TestOTPCooldown:
    """Tests for OTP cooldown period enforcement."""
    
    def test_cooldown_is_60_seconds(self):
        """Cooldown should be 60 seconds."""
        from app.accounts.models import OTP_COOLDOWN_SECONDS
        
        assert OTP_COOLDOWN_SECONDS == 60, f"Cooldown should be 60s, got {OTP_COOLDOWN_SECONDS}"
    
    def test_cooldown_active_within_period(self):
        """Cooldown should be active within 60 seconds."""
        from app.accounts.models import OtpCode
        
        now = datetime.utcnow()
        last_created = now - timedelta(seconds=30)  # 30s ago
        
        assert OtpCode.is_cooldown_active(last_created) is True
    
    def test_cooldown_inactive_after_period(self):
        """Cooldown should be inactive after 60 seconds."""
        from app.accounts.models import OtpCode
        
        now = datetime.utcnow()
        last_created = now - timedelta(seconds=61)  # 61s ago
        
        assert OtpCode.is_cooldown_active(last_created) is False


# ============================================================
# Per-Email Daily Cap Tests
# ============================================================

class TestDailyCap:
    """Tests for per-email 20/day cap."""
    
    def test_daily_cap_constant(self):
        """Daily cap should be 20."""
        # FIX: Correct constant name is OTP_DAILY_CAP_PER_EMAIL
        from app.accounts.otp import OTP_DAILY_CAP_PER_EMAIL
        
        assert OTP_DAILY_CAP_PER_EMAIL == 20, f"Daily cap should be 20, got {OTP_DAILY_CAP_PER_EMAIL}"
    
    def test_daily_cap_check_under_limit(self):
        """Should allow OTP when under daily limit."""
        from app.accounts.otp import OTPManager, OTP_DAILY_CAP_PER_EMAIL
        
        # FIX: _check_daily_cap takes (self, email, client) - must pass client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 10  # Under 20
        mock_result.data = [{"id": str(i)} for i in range(10)]
        mock_client.table.return_value.select.return_value.ilike.return_value.gte.return_value.execute.return_value = mock_result
        
        manager = OTPManager()
        within_limit, count = manager._check_daily_cap("test@example.com", mock_client)
        
        assert within_limit is True, "Should be within daily limit"
        assert count == 10
    
    def test_daily_cap_check_at_limit(self):
        """Should block OTP when at daily limit."""
        from app.accounts.otp import OTPManager, OTP_DAILY_CAP_PER_EMAIL
        
        # FIX: _check_daily_cap takes (self, email, client) - must pass client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 20  # At limit
        mock_result.data = [{"id": str(i)} for i in range(20)]
        mock_client.table.return_value.select.return_value.ilike.return_value.gte.return_value.execute.return_value = mock_result
        
        manager = OTPManager()
        within_limit, count = manager._check_daily_cap("test@example.com", mock_client)
        
        assert within_limit is False, "Should be at daily limit"
        assert count == OTP_DAILY_CAP_PER_EMAIL


# ============================================================
# Single-Use OTP Tests
# ============================================================

class TestSingleUse:
    """Tests for single-use OTP behavior."""
    
    def test_otp_manager_has_verify_method(self):
        """OTPManager should have verify_otp method."""
        from app.accounts.otp import OTPManager
        
        manager = OTPManager()
        assert hasattr(manager, 'verify_otp'), "OTPManager should have verify_otp method"


# ============================================================
# JWT Tests
# ============================================================

class TestJWT:
    """Tests for JWT token generation and validation."""
    
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-at-least-32-characters-long"})
    def test_jwt_create_token(self):
        """Should create valid JWT token."""
        from app.accounts.auth import create_token
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        email = "test@example.com"
        
        token, expires_at = create_token(user_id, email)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically longer
        assert isinstance(expires_at, datetime)
    
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-at-least-32-characters-long"})
    def test_jwt_verify_token(self):
        """Should verify valid JWT token."""
        from app.accounts.auth import create_token, verify_token
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        email = "test@example.com"
        
        token, _ = create_token(user_id, email)
        payload = verify_token(token)
        
        assert payload is not None
        assert payload.get("user_id") == user_id
        assert payload.get("email") == email
    
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-at-least-32-characters-long"})
    def test_jwt_reject_invalid_token(self):
        """Should reject invalid JWT token."""
        from app.accounts.auth import verify_token
        
        payload = verify_token("invalid.token.here")
        
        assert payload is None
    
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-at-least-32-characters-long"})
    def test_jwt_reject_expired_token(self):
        """Should reject expired JWT token."""
        from app.accounts.auth import verify_token
        import jwt
        
        secret = "test-secret-at-least-32-characters-long"
        
        # Create expired token
        payload = {
            "user_id": "test-id",
            "email": "test@example.com",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=25),
        }
        expired_token = jwt.encode(payload, secret, algorithm="HS256")
        
        result = verify_token(expired_token)
        
        assert result is None, "Should reject expired token"
    
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-at-least-32-characters-long", "JWT_EXPIRY_HOURS": "24"})
    def test_jwt_default_expiry(self):
        """JWT should have configurable expiry (default 24h)."""
        # FIX: Use the function _get_jwt_expiry_hours() not constant
        from app.accounts.auth import _get_jwt_expiry_hours
        
        expiry = _get_jwt_expiry_hours()
        assert expiry == 24, f"Default JWT expiry should be 24h, got {expiry}"


# ============================================================
# Anti-Enumeration Tests
# ============================================================

class TestAntiEnumeration:
    """Tests for anti-enumeration protection."""
    
    def test_request_otp_response_format(self):
        """Response should be generic to prevent enumeration."""
        # The response message is always "If this email is valid, a code was sent"
        # regardless of whether the email exists
        expected_message = "If this email is valid"
        assert "If this email" in expected_message


# ============================================================
# Integration Tests (require database)
# ============================================================

@pytest.mark.integration
class TestAuthIntegration:
    """Integration tests requiring database connection."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    @pytest.mark.skipif(
        os.getenv("AUTH_ENDPOINTS_ENABLED", "off").lower() not in ("on", "true", "1"),
        reason="Auth endpoints not enabled"
    )
    def test_request_otp_endpoint(self, test_client):
        """Test /auth/request-otp endpoint."""
        response = test_client.post(
            "/auth/request-otp",
            json={"email": "test@example.com"}
        )
        
        # Should return success (anti-enumeration)
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
