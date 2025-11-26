-- migrations/001_phase_3_1.sql
-- Phase 3.1: User Accounts & Memory (Foundations)
-- 
-- Run this script manually in Supabase SQL Editor.
-- No Alembic/migration tool — raw SQL only per infrastructure decision.
--
-- Tables (4):
--   1. users           — User accounts with optional memory opt-in
--   2. orders          — Order history (user_id nullable for guests)
--   3. recipient_profiles — Saved recipient preferences
--   4. otp_codes       — OTP storage for auth and privacy verification
--
-- Indexes:
--   - UNIQUE INDEX on users.email
--   - INDEX on orders.user_id
--   - INDEX on orders.created_at
--   - INDEX on orders.email (for historical guest linkage)
--   - INDEX on recipient_profiles.user_id
--   - INDEX on otp_codes.email
--
-- Privacy Rails:
--   - No PII in preferences (enforced at application layer)
--   - Retention job de-identifies orders after 90 days for non-opt-in users
--   - OTP codes are hashed (SHA-256) before storage

-- ============================================================
-- Enable UUID extension (if not already enabled)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Table 1: users
-- ============================================================
-- User accounts with optional memory opt-in.
-- Guest checkout creates orders without user_id.
-- Soft accounts: email/phone captured, optional login.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL,
    phone TEXT,  -- E.164 format: +911234567890
    memory_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- UNIQUE INDEX on email (case-insensitive matching at application layer)
-- This is the PRIMARY INDEX for user lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique 
    ON users (LOWER(email));

-- Comment for documentation
COMMENT ON TABLE users IS 'User accounts for Phase 3.1. memory_opt_in controls retention.';
COMMENT ON COLUMN users.email IS 'Primary identifier. Unique, case-insensitive.';
COMMENT ON COLUMN users.phone IS 'Optional. E.164 format (+911234567890).';
COMMENT ON COLUMN users.memory_opt_in IS 'If true, retention extended beyond 90 days.';

-- ============================================================
-- Table 2: orders
-- ============================================================
-- Order history. user_id is nullable for guest checkout.
-- email column enables 90-day historical guest linkage on opt-in.

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    email TEXT,  -- For 90-day historical guest linkage
    sku_id TEXT NOT NULL,
    emotion TEXT NOT NULL,  -- Frozen anchor enum
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- INDEX on user_id for user order history queries
CREATE INDEX IF NOT EXISTS idx_orders_user_id 
    ON orders (user_id) 
    WHERE user_id IS NOT NULL;

-- INDEX on created_at for retention job and time-based queries
CREATE INDEX IF NOT EXISTS idx_orders_created_at 
    ON orders (created_at);

-- INDEX on email for historical guest linkage
CREATE INDEX IF NOT EXISTS idx_orders_email 
    ON orders (LOWER(email)) 
    WHERE email IS NOT NULL;

-- Comment for documentation
COMMENT ON TABLE orders IS 'Order history. user_id nullable for guest checkout.';
COMMENT ON COLUMN orders.user_id IS 'FK to users. NULL for guest orders.';
COMMENT ON COLUMN orders.email IS 'For 90-day historical guest linkage on opt-in.';
COMMENT ON COLUMN orders.sku_id IS 'Product SKU from catalog.json.';
COMMENT ON COLUMN orders.emotion IS 'Emotion anchor (frozen enum from Phase 1.6).';

-- ============================================================
-- Table 3: recipient_profiles
-- ============================================================
-- Saved recipient preferences for authenticated users.
-- user_id is required (NOT NULL) — profiles belong to logged-in users.

CREATE TABLE IF NOT EXISTS recipient_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,  -- e.g., "Mom", "Dad"
    preferences JSONB,   -- Validated at application layer
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- INDEX on user_id for profile listing
CREATE INDEX IF NOT EXISTS idx_recipient_profiles_user_id 
    ON recipient_profiles (user_id);

-- Comment for documentation
COMMENT ON TABLE recipient_profiles IS 'Saved recipient preferences. Belongs to authenticated user.';
COMMENT ON COLUMN recipient_profiles.name IS 'Recipient name (e.g., Mom, Dad).';
COMMENT ON COLUMN recipient_profiles.preferences IS 'JSONB with whitelisted keys: flowers, tier, palette_pref.';
COMMENT ON COLUMN recipient_profiles.schema_version IS 'Preferences schema version for migration.';

-- ============================================================
-- Table 4: otp_codes
-- ============================================================
-- OTP storage for authentication and privacy verification.
-- Security: OTP hashed with SHA-256 before storage.

CREATE TABLE IF NOT EXISTS otp_codes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL,
    otp_hash TEXT NOT NULL,  -- SHA-256 hash of 6-digit OTP
    expires_at TIMESTAMPTZ NOT NULL,  -- created_at + 10 minutes
    attempts INTEGER NOT NULL DEFAULT 0,  -- Max 5
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- INDEX on email for OTP lookup
CREATE INDEX IF NOT EXISTS idx_otp_codes_email 
    ON otp_codes (LOWER(email));

-- INDEX on expires_at for cleanup job
CREATE INDEX IF NOT EXISTS idx_otp_codes_expires_at 
    ON otp_codes (expires_at);

-- Comment for documentation
COMMENT ON TABLE otp_codes IS 'OTP storage for auth and privacy verification.';
COMMENT ON COLUMN otp_codes.otp_hash IS 'SHA-256 hash of 6-digit OTP. Never store plaintext.';
COMMENT ON COLUMN otp_codes.expires_at IS 'created_at + 10 minutes.';
COMMENT ON COLUMN otp_codes.attempts IS 'Verification attempts. Max 5 before lockout.';

-- ============================================================
-- Retention Policy Helper View (Optional)
-- ============================================================
-- View to identify orders eligible for de-identification.
-- Used by retention_job.py.

CREATE OR REPLACE VIEW orders_eligible_for_retention AS
SELECT o.id, o.user_id, o.email, o.created_at
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
WHERE o.created_at < NOW() - INTERVAL '90 days'
  AND (
      o.user_id IS NULL  -- Guest orders always eligible
      OR u.memory_opt_in = FALSE  -- Non-opt-in users
  );

COMMENT ON VIEW orders_eligible_for_retention IS 
    'Orders eligible for de-identification (90-day retention). Used by retention job.';

-- ============================================================
-- Cleanup Function for Expired OTPs (Optional)
-- ============================================================
-- Can be called periodically or used with pg_cron.

CREATE OR REPLACE FUNCTION cleanup_expired_otps()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM otp_codes
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_expired_otps IS 
    'Deletes expired OTP codes. Returns count of deleted rows.';

-- ============================================================
-- Migration Metadata
-- ============================================================
-- Track migration version for auditing.

CREATE TABLE IF NOT EXISTS _schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

INSERT INTO _schema_migrations (version, description)
VALUES ('001_phase_3_1', 'Phase 3.1: User Accounts & Memory foundations')
ON CONFLICT (version) DO NOTHING;

-- ============================================================
-- Verification Queries (Run after migration to verify)
-- ============================================================
-- 
-- Check tables exist:
--   SELECT table_name FROM information_schema.tables 
--   WHERE table_schema = 'public' AND table_name IN ('users', 'orders', 'recipient_profiles', 'otp_codes');
--
-- Check indexes exist:
--   SELECT indexname FROM pg_indexes 
--   WHERE schemaname = 'public' AND tablename IN ('users', 'orders', 'recipient_profiles', 'otp_codes');
--
-- Check foreign keys:
--   SELECT conname, conrelid::regclass, confrelid::regclass
--   FROM pg_constraint WHERE contype = 'f';

-- ============================================================
-- END OF MIGRATION 001_phase_3_1
-- ============================================================
