-- Run this in the Supabase SQL editor to enable enhanced analytics tracking.
-- Safe to run multiple times (uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

-- 1. Replace ip_hash with visitor_id (client-side UUID — no IP stored)
ALTER TABLE page_views ADD COLUMN IF NOT EXISTS visitor_id TEXT;
ALTER TABLE chat_events ADD COLUMN IF NOT EXISTS visitor_id TEXT;

-- 2. New enrichment columns on page_views
ALTER TABLE page_views ADD COLUMN IF NOT EXISTS device_type TEXT;   -- 'mobile' | 'tablet' | 'desktop'
ALTER TABLE page_views ADD COLUMN IF NOT EXISTS country     TEXT;   -- ISO 3166-1 alpha-2, e.g. 'IN', 'US'
ALTER TABLE page_views ADD COLUMN IF NOT EXISTS referrer    TEXT;   -- hostname only, e.g. 'twitter.com' or NULL for direct

-- 3. Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_page_views_visitor_id   ON page_views (visitor_id);
CREATE INDEX IF NOT EXISTS idx_page_views_created_at   ON page_views (created_at);
CREATE INDEX IF NOT EXISTS idx_page_views_country      ON page_views (country);
CREATE INDEX IF NOT EXISTS idx_page_views_device_type  ON page_views (device_type);
CREATE INDEX IF NOT EXISTS idx_chat_events_visitor_id  ON chat_events (visitor_id);
