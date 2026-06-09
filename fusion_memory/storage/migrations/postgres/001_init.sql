-- Fusion Memory Postgres/pgvector schema.
-- Mirrors /home/wwb/fusion-memory-execution-plan.md for production backends.

create extension if not exists vector;

create table if not exists evidence_spans (
  span_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  turn_id text,
  speaker text not null,
  span_type text not null,
  content text not null,
  content_hash text not null,
  timestamp timestamptz,
  source_uri text,
  line_start int,
  line_end int,
  parent_span_id text,
  entities jsonb not null default '[]',
  topics jsonb not null default '[]',
  embedding_dense vector(1024),
  search_tsv tsvector generated always as (to_tsvector('simple', content)) stored,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists evidence_scope_idx on evidence_spans(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists evidence_timestamp_idx on evidence_spans(timestamp);
create index if not exists evidence_hash_idx on evidence_spans(content_hash);
create index if not exists evidence_search_idx on evidence_spans using gin(search_tsv);
create index if not exists evidence_embedding_idx on evidence_spans using hnsw (embedding_dense vector_cosine_ops);

create table if not exists memory_facts (
  fact_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  subject text,
  predicate text,
  object text,
  text text not null,
  category text not null,
  polarity text not null default 'unknown',
  confidence double precision not null,
  salience double precision not null,
  observed_at timestamptz,
  valid_from timestamptz,
  valid_to timestamptz,
  source_span_ids jsonb not null default '[]',
  linked_fact_ids jsonb not null default '[]',
  embedding_dense vector(1024),
  search_tsv tsvector generated always as (to_tsvector('simple', text)) stored,
  hash text,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists facts_scope_idx on memory_facts(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists facts_category_idx on memory_facts(category);
create index if not exists facts_valid_idx on memory_facts(valid_from, valid_to);
create index if not exists facts_search_idx on memory_facts using gin(search_tsv);
create index if not exists facts_embedding_idx on memory_facts using hnsw (embedding_dense vector_cosine_ops);

create table if not exists fact_relations (
  relation_id text primary key,
  from_fact_id text not null references memory_facts(fact_id) on delete cascade,
  to_fact_id text not null references memory_facts(fact_id) on delete cascade,
  relation_type text not null,
  source_span_ids jsonb not null default '[]',
  confidence double precision not null,
  created_at timestamptz not null default now()
);

create index if not exists fact_rel_from_idx on fact_relations(from_fact_id);
create index if not exists fact_rel_to_idx on fact_relations(to_fact_id);
create index if not exists fact_rel_type_idx on fact_relations(relation_type);

create table if not exists events (
  event_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  event_type text not null,
  participants jsonb not null default '[]',
  description text not null,
  time_start timestamptz,
  time_end timestamptz,
  time_granularity text,
  time_source text,
  source_span_ids jsonb not null default '[]',
  fact_ids jsonb not null default '[]',
  confidence double precision not null,
  search_tsv tsvector generated always as (to_tsvector('simple', description)) stored,
  created_at timestamptz not null default now()
);

create index if not exists events_scope_idx on events(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists events_time_idx on events(time_start, time_end);
create index if not exists events_type_idx on events(event_type);
create index if not exists events_search_idx on events using gin(search_tsv);

create table if not exists event_edges (
  edge_id text primary key,
  from_event_id text not null references events(event_id) on delete cascade,
  to_event_id text not null references events(event_id) on delete cascade,
  edge_type text not null,
  source_span_ids jsonb not null default '[]',
  confidence double precision not null,
  created_at timestamptz not null default now()
);

create index if not exists event_edges_from_idx on event_edges(from_event_id);
create index if not exists event_edges_to_idx on event_edges(to_event_id);
create index if not exists event_edges_type_idx on event_edges(edge_type);

create table if not exists current_views (
  view_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  view_type text not null,
  subject text not null,
  text text not null,
  state_json jsonb not null default '{}',
  source_fact_ids jsonb not null default '[]',
  source_event_ids jsonb not null default '[]',
  source_span_ids jsonb not null default '[]',
  confidence double precision not null,
  updated_at timestamptz not null default now(),
  expires_at timestamptz
);

create index if not exists current_views_scope_idx on current_views(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists current_views_type_idx on current_views(view_type);

create table if not exists entity_profiles (
  profile_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  entity_id text not null,
  entity_type text not null,
  profile_type text not null,
  text text not null,
  state_json jsonb not null default '{}',
  source_fact_ids jsonb not null default '[]',
  source_event_ids jsonb not null default '[]',
  source_span_ids jsonb not null default '[]',
  confidence double precision not null,
  support_count int not null default 1,
  last_observed_at timestamptz,
  updated_at timestamptz not null default now(),
  expires_at timestamptz,
  embedding_dense vector(1024),
  search_tsv tsvector generated always as (to_tsvector('simple', text)) stored
);

create index if not exists profiles_scope_idx on entity_profiles(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists profiles_entity_idx on entity_profiles(entity_id, profile_type);
create index if not exists profiles_search_idx on entity_profiles using gin(search_tsv);
create index if not exists profiles_embedding_idx on entity_profiles using hnsw (embedding_dense vector_cosine_ops);

create table if not exists entities (
  entity_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  name text not null,
  entity_type text not null,
  aliases jsonb not null default '[]',
  source_span_ids jsonb not null default '[]',
  observed_count int not null default 1,
  last_observed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists entities_scope_idx on entities(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists entities_name_idx on entities(lower(name));

create table if not exists encoding_decisions (
  decision_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  candidate_type text not null,
  candidate_json jsonb not null,
  source_span_ids jsonb not null default '[]',
  decision text not null,
  reason_codes jsonb not null default '[]',
  scores_json jsonb not null default '{}',
  matched_existing_ids jsonb not null default '[]',
  created_at timestamptz not null default now()
);

create index if not exists encoding_scope_idx on encoding_decisions(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists encoding_decision_idx on encoding_decisions(decision);

create table if not exists retrieval_utility_examples (
  example_id text primary key,
  query_id text,
  query_text text not null,
  query_type text,
  candidate_id text not null,
  candidate_type text not null,
  features_json jsonb not null,
  label text not null,
  label_source text not null,
  answer_correct boolean,
  created_at timestamptz not null default now()
);

create index if not exists utility_query_type_idx on retrieval_utility_examples(query_type);
create index if not exists utility_label_idx on retrieval_utility_examples(label);

create table if not exists debug_traces (
  trace_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  trace_json jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists debug_traces_scope_idx on debug_traces(workspace_id, user_id, agent_id, run_id, session_id);

create table if not exists audit_events (
  audit_id text primary key,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  event_type text not null,
  object_type text,
  object_id text,
  trace_id text,
  payload_json jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists audit_scope_idx on audit_events(workspace_id, user_id, agent_id, run_id, session_id);
create index if not exists audit_trace_idx on audit_events(trace_id);

create table if not exists background_tasks (
  task_id text primary key,
  task_type text not null,
  workspace_id text,
  user_id text,
  agent_id text,
  run_id text,
  session_id text,
  app_id text,
  status text not null,
  dedupe_key text,
  payload_json jsonb not null default '{}',
  attempts int not null default 0,
  last_error text,
  run_after timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists background_tasks_dedupe_idx on background_tasks(dedupe_key) where dedupe_key is not null;
create index if not exists background_tasks_status_idx on background_tasks(status, run_after, created_at);
create index if not exists background_tasks_scope_idx on background_tasks(workspace_id, user_id, agent_id, run_id, session_id);
