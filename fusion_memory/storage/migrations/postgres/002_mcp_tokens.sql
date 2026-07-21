create table if not exists memory_api_tokens (
  token_id text primary key,
  token_hash text not null,
  user_id text not null,
  scopes jsonb not null default '[]',
  expires_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz not null default now(),
  last_used_at timestamptz
);

create unique index if not exists memory_api_tokens_token_id_idx on memory_api_tokens(token_id);
create index if not exists memory_api_tokens_token_hash_idx on memory_api_tokens(token_hash);
create index if not exists memory_api_tokens_user_id_idx on memory_api_tokens(user_id);
