
-- Migration: 20251103071043
-- Create archive table for storing generated papers and code
CREATE TABLE public.archive (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('paper', 'code')),
  concept TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.archive ENABLE ROW LEVEL SECURITY;

-- Create policy to allow anyone to view archives (public access)
CREATE POLICY "Anyone can view archives"
ON public.archive
FOR SELECT
USING (true);

-- Create policy to allow anyone to insert archives (public access)
CREATE POLICY "Anyone can create archives"
ON public.archive
FOR INSERT
WITH CHECK (true);

-- Create index for faster searches
CREATE INDEX idx_archive_title ON public.archive(title);
CREATE INDEX idx_archive_type ON public.archive(type);
CREATE INDEX idx_archive_created_at ON public.archive(created_at DESC);

-- Migration: 20251103075343
-- Remove the archive table
DROP TABLE IF EXISTS public.archive CASCADE;
