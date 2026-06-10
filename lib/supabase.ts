import { createClient } from '@supabase/supabase-js'

// Anon client — insert-only once RLS is enabled. Used by /api/track.
export const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_ANON_KEY!
)

// Admin client — bypasses RLS, used by /api/analytics for reads.
// Falls back to the anon key so analytics keeps working on deployments
// where SUPABASE_SERVICE_ROLE_KEY isn't set yet (pre-RLS).
export const supabaseAdmin = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY!
)
