-- ============================================================================
-- HealOnCal Complete Supabase Database Setup
-- ============================================================================
-- Run this SQL in your Supabase SQL Editor to create the complete setup
-- for the skin analysis application with storage, tables, policies, and functions

-- ============================================================================
-- 1. STORAGE BUCKET SETUP
-- ============================================================================

-- Create the storage bucket for skin scan images
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'skin-scans',
    'skin-scans', 
    true,
    10485760, -- 10MB limit
    ARRAY['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 2. STORAGE POLICIES
-- ============================================================================

-- Policy: Allow authenticated users to upload files
CREATE POLICY "Allow authenticated uploads" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (bucket_id = 'skin-scans');

-- Policy: Allow public access to view files
CREATE POLICY "Allow public downloads" ON storage.objects
FOR SELECT TO public
USING (bucket_id = 'skin-scans');

-- Policy: Allow service role to manage all files
CREATE POLICY "Allow service role full access" ON storage.objects
FOR ALL TO service_role
USING (bucket_id = 'skin-scans')
WITH CHECK (bucket_id = 'skin-scans');

-- Policy: Allow authenticated users to update their own files
CREATE POLICY "Allow authenticated updates" ON storage.objects
FOR UPDATE TO authenticated
USING (bucket_id = 'skin-scans')
WITH CHECK (bucket_id = 'skin-scans');

-- Policy: Allow authenticated users to delete their own files
CREATE POLICY "Allow authenticated deletes" ON storage.objects
FOR DELETE TO authenticated
USING (bucket_id = 'skin-scans');

-- ============================================================================
-- 3. MAIN TABLES
-- ============================================================================

-- Create the skin_analyses table
CREATE TABLE IF NOT EXISTS public.skin_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id TEXT NOT NULL,
    angle TEXT NOT NULL CHECK (angle IN ('front', 'left', 'right')),
    image_url TEXT NOT NULL,
    analysis JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Additional useful columns
    session_id TEXT,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    analysis_status TEXT DEFAULT 'completed' CHECK (analysis_status IN ('pending', 'processing', 'completed', 'failed')),
    file_size INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    processing_time_ms INTEGER,
    
    -- Constraints
    CONSTRAINT unique_client_angle_session UNIQUE (client_id, angle, session_id)
);

-- Create analysis sessions table for grouping multi-angle captures
CREATE TABLE IF NOT EXISTS public.analysis_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_status TEXT DEFAULT 'active' CHECK (session_status IN ('active', 'completed', 'abandoned')),
    total_captures INTEGER DEFAULT 0,
    completed_angles TEXT[] DEFAULT ARRAY[]::TEXT[],
    remaining_angles TEXT[] DEFAULT ARRAY['front', 'left', 'right'],
    final_analysis JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Create user preferences table
CREATE TABLE IF NOT EXISTS public.user_preferences (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
    skin_type TEXT,
    skin_concerns TEXT[],
    preferred_language TEXT DEFAULT 'en',
    notification_preferences JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create analysis history summary table for quick lookups
CREATE TABLE IF NOT EXISTS public.analysis_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
    analysis_date DATE DEFAULT CURRENT_DATE,
    skin_type TEXT,
    dominant_concerns TEXT[],
    overall_score DECIMAL(3,2),
    improvement_score DECIMAL(3,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 4. INDEXES FOR PERFORMANCE
-- ============================================================================

-- Indexes for skin_analyses table
CREATE INDEX IF NOT EXISTS idx_skin_analyses_client_id ON public.skin_analyses(client_id);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_created_at ON public.skin_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_angle ON public.skin_analyses(angle);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_session_id ON public.skin_analyses(session_id);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_user_id ON public.skin_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_status ON public.skin_analyses(analysis_status);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_skin_analyses_client_angle ON public.skin_analyses(client_id, angle);
CREATE INDEX IF NOT EXISTS idx_skin_analyses_user_date ON public.skin_analyses(user_id, created_at DESC);

-- Indexes for analysis_sessions table
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_client_id ON public.analysis_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON public.analysis_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON public.analysis_sessions(session_status);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON public.analysis_sessions(created_at DESC);

-- Indexes for user_preferences table
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON public.user_preferences(user_id);

-- Indexes for analysis_history table
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_id ON public.analysis_history(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_history_date ON public.analysis_history(analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_date ON public.analysis_history(user_id, analysis_date DESC);

-- ============================================================================
-- 5. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE public.skin_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies for skin_analyses table
CREATE POLICY "Users can insert their own skin analyses" ON public.skin_analyses
    FOR INSERT TO authenticated
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can view their own skin analyses" ON public.skin_analyses
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own skin analyses" ON public.skin_analyses
    FOR UPDATE TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own skin analyses" ON public.skin_analyses
    FOR DELETE TO authenticated
    USING (auth.uid() = user_id);

-- Service role can manage all skin analyses
CREATE POLICY "Service role can manage all skin analyses" ON public.skin_analyses
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- RLS Policies for analysis_sessions table
CREATE POLICY "Users can manage their own sessions" ON public.analysis_sessions
    FOR ALL TO authenticated
    USING (auth.uid() = user_id OR user_id IS NULL)
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Service role can manage all sessions" ON public.analysis_sessions
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- RLS Policies for user_preferences table
CREATE POLICY "Users can manage their own preferences" ON public.user_preferences
    FOR ALL TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Service role can manage all preferences" ON public.user_preferences
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- RLS Policies for analysis_history table
CREATE POLICY "Users can view their own history" ON public.analysis_history
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage all history" ON public.analysis_history
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- 6. TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to automatically update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating updated_at columns
CREATE TRIGGER update_skin_analyses_updated_at
    BEFORE UPDATE ON public.skin_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analysis_sessions_updated_at
    BEFORE UPDATE ON public.analysis_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON public.user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to update session progress when analysis is added
CREATE OR REPLACE FUNCTION update_session_progress()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the session with the new capture
    UPDATE public.analysis_sessions 
    SET 
        total_captures = total_captures + 1,
        completed_angles = array_append(completed_angles, NEW.angle),
        remaining_angles = array_remove(remaining_angles, NEW.angle),
        updated_at = NOW()
    WHERE id::text = NEW.session_id;
    
    -- Mark session as completed if all angles are done
    UPDATE public.analysis_sessions 
    SET 
        session_status = 'completed',
        completed_at = NOW()
    WHERE id::text = NEW.session_id 
    AND array_length(remaining_angles, 1) = 0;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to update session progress
CREATE TRIGGER update_session_on_analysis_insert
    AFTER INSERT ON public.skin_analyses
    FOR EACH ROW
    WHEN (NEW.session_id IS NOT NULL)
    EXECUTE FUNCTION update_session_progress();

-- ============================================================================
-- 7. UTILITY FUNCTIONS
-- ============================================================================

-- Function to get latest analysis for a client
CREATE OR REPLACE FUNCTION get_latest_analysis(client_uuid TEXT, max_results INTEGER DEFAULT 3)
RETURNS TABLE (
    id UUID,
    client_id TEXT,
    angle TEXT,
    image_url TEXT,
    analysis JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT sa.id, sa.client_id, sa.angle, sa.image_url, sa.analysis, sa.created_at
    FROM public.skin_analyses sa
    WHERE sa.client_id = client_uuid
    ORDER BY sa.created_at DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's analysis history with summary
CREATE OR REPLACE FUNCTION get_user_analysis_summary(user_uuid UUID, days_back INTEGER DEFAULT 30)
RETURNS TABLE (
    analysis_count BIGINT,
    latest_analysis TIMESTAMPTZ,
    skin_types_detected TEXT[],
    common_concerns TEXT[],
    average_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as analysis_count,
        MAX(sa.created_at) as latest_analysis,
        array_agg(DISTINCT sa.analysis->>'skin_type') FILTER (WHERE sa.analysis->>'skin_type' IS NOT NULL) as skin_types_detected,
        array_agg(DISTINCT concern) FILTER (WHERE concern IS NOT NULL) as common_concerns,
        AVG((sa.analysis->>'overall_score')::decimal) as average_score
    FROM public.skin_analyses sa
    CROSS JOIN LATERAL jsonb_array_elements_text(
        CASE WHEN jsonb_typeof(sa.analysis->'concerns') = 'array' 
             THEN sa.analysis->'concerns'
             ELSE '[]'::jsonb
        END
    ) AS concern
    WHERE sa.user_id = user_uuid
    AND sa.created_at >= NOW() - INTERVAL '1 day' * days_back;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clean up old temporary sessions
CREATE OR REPLACE FUNCTION cleanup_abandoned_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Mark sessions as abandoned if they're older than 24 hours and still active
    UPDATE public.analysis_sessions 
    SET session_status = 'abandoned'
    WHERE session_status = 'active' 
    AND created_at < NOW() - INTERVAL '24 hours';
    
    -- Delete analysis records for abandoned sessions older than 7 days
    DELETE FROM public.skin_analyses 
    WHERE session_id IN (
        SELECT id::text FROM public.analysis_sessions 
        WHERE session_status = 'abandoned' 
        AND created_at < NOW() - INTERVAL '7 days'
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete the abandoned sessions
    DELETE FROM public.analysis_sessions 
    WHERE session_status = 'abandoned' 
    AND created_at < NOW() - INTERVAL '7 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- 8. VIEWS FOR EASIER QUERYING
-- ============================================================================

-- View for analysis summary with computed fields
CREATE OR REPLACE VIEW public.skin_analyses_summary AS
SELECT 
    sa.id,
    sa.client_id,
    sa.user_id,
    sa.session_id,
    sa.angle,
    sa.image_url,
    sa.analysis_status,
    sa.analysis->>'skin_type' as skin_type,
    sa.analysis->>'skin_tone' as skin_tone,
    (sa.analysis->>'perceived_age')::integer as perceived_age,
    (sa.analysis->>'moisture_level')::decimal as moisture_level,
    (sa.analysis->>'oiliness')::decimal as oiliness,
    sa.analysis->'concerns' as concerns,
    sa.analysis->'recommendations' as recommendations,
    sa.created_at,
    sa.updated_at
FROM public.skin_analyses sa;

-- View for complete session information
CREATE OR REPLACE VIEW public.complete_sessions AS
SELECT 
    sess.id as session_id,
    sess.client_id,
    sess.user_id,
    sess.session_status,
    sess.total_captures,
    sess.completed_angles,
    sess.remaining_angles,
    sess.final_analysis,
    sess.created_at as session_created,
    sess.completed_at,
    COALESCE(
        json_agg(
            json_build_object(
                'id', sa.id,
                'angle', sa.angle,
                'image_url', sa.image_url,
                'analysis', sa.analysis,
                'created_at', sa.created_at
            ) ORDER BY sa.created_at
        ) FILTER (WHERE sa.id IS NOT NULL),
        '[]'::json
    ) as analyses
FROM public.analysis_sessions sess
LEFT JOIN public.skin_analyses sa ON sess.id::text = sa.session_id
GROUP BY sess.id, sess.client_id, sess.user_id, sess.session_status, 
         sess.total_captures, sess.completed_angles, sess.remaining_angles, 
         sess.final_analysis, sess.created_at, sess.completed_at;

-- ============================================================================
-- 9. GRANT PERMISSIONS
-- ============================================================================

-- Grant permissions on tables
GRANT ALL ON public.skin_analyses TO authenticated, service_role;
GRANT ALL ON public.analysis_sessions TO authenticated, service_role;
GRANT ALL ON public.user_preferences TO authenticated, service_role;
GRANT ALL ON public.analysis_history TO authenticated, service_role;

-- Grant permissions on views
GRANT SELECT ON public.skin_analyses_summary TO authenticated, service_role;
GRANT SELECT ON public.complete_sessions TO authenticated, service_role;

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION get_latest_analysis(TEXT, INTEGER) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION get_user_analysis_summary(UUID, INTEGER) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION cleanup_abandoned_sessions() TO service_role;

-- ============================================================================
-- 10. INITIAL DATA AND CONFIGURATIONS
-- ============================================================================

-- Create a scheduled job to cleanup abandoned sessions (requires pg_cron extension)
-- Uncomment the following lines if you have pg_cron enabled:
-- SELECT cron.schedule('cleanup-abandoned-sessions', '0 2 * * *', 'SELECT cleanup_abandoned_sessions();');

-- Insert default user preferences for skin types (if needed)
-- Uncomment and modify the following lines to add default preferences:
-- INSERT INTO public.user_preferences (user_id, skin_type, skin_concerns) 
-- VALUES 
-- (uuid_generate_v4(), 'combination', ARRAY['acne', 'oiliness']),
-- (uuid_generate_v4(), 'dry', ARRAY['wrinkles', 'dryness'])
-- ON CONFLICT DO NOTHING;

-- ============================================================================
-- SETUP COMPLETE
-- ============================================================================

-- Verify the setup
DO $$
BEGIN
    RAISE NOTICE 'HealOnCal Supabase setup completed successfully!';
    RAISE NOTICE 'Tables created: % tables', (
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('skin_analyses', 'analysis_sessions', 'user_preferences', 'analysis_history')
    );
    RAISE NOTICE 'Storage bucket created: skin-scans';
    RAISE NOTICE 'Views created: % views', (
        SELECT COUNT(*) FROM information_schema.views 
        WHERE table_schema = 'public' 
        AND table_name IN ('skin_analyses_summary', 'complete_sessions')
    );
    RAISE NOTICE 'Functions created: % functions', (
        SELECT COUNT(*) FROM information_schema.routines 
        WHERE routine_schema = 'public' 
        AND routine_name IN ('get_latest_analysis', 'get_user_analysis_summary', 'cleanup_abandoned_sessions')
    );
END $$; 