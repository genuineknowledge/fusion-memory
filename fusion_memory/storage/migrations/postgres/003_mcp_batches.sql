create table if not exists fusion_memory_mcp_batches (
  user_id text not null,
  batch_id text not null,
  request_hash text not null,
  status text not null default 'pending' check (status in ('pending', 'completed')),
  result jsonb,
  trace_id text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz,
  primary key (user_id, batch_id)
);

create index if not exists fusion_memory_mcp_batches_status_idx
  on fusion_memory_mcp_batches (user_id, status);
create index if not exists fusion_memory_mcp_batches_trace_idx
  on fusion_memory_mcp_batches (trace_id);
