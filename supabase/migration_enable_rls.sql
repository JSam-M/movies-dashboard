-- Enable Row Level Security on the analytics tables.
-- Run this in the Supabase SQL editor AFTER setting SUPABASE_SERVICE_ROLE_KEY
-- in Vercel and redeploying — the analytics API reads with the service role
-- key, which bypasses RLS. The anon key keeps insert-only access for tracking.
-- Safe to run multiple times.

ALTER TABLE page_views  ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_events ENABLE ROW LEVEL SECURITY;

-- Tracking inserts (via /api/track using the anon key) stay allowed
DROP POLICY IF EXISTS "anon can insert page views" ON page_views;
CREATE POLICY "anon can insert page views"
  ON page_views FOR INSERT TO anon
  WITH CHECK (true);

DROP POLICY IF EXISTS "anon can insert chat events" ON chat_events;
CREATE POLICY "anon can insert chat events"
  ON chat_events FOR INSERT TO anon
  WITH CHECK (true);

-- No SELECT/UPDATE/DELETE policies for anon: with RLS enabled and no policy,
-- those operations are denied. Reads happen via the service_role key only.
