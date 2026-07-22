# Retrieval Heuristic Inventory

This inventory classifies the legacy retrieval surface before the product retrieval engine refactor. It covers every `MemoryService` method in `fusion_memory/api/service.py:1640-3737` and every top-level function in `fusion_memory/api/service_helpers.py`.

| Legacy group | Destination | Reason |
| --- | --- | --- |
| raw/fact/event/view/profile hybrid recall | Product providers | Generic product retrieval capability |
| `_entity_candidates`, simplified `_exact_candidates` | Product entity/lexical providers | Explicit entity and lexical behavior |
| persisted chronology selector | Product chronology provider | Generic ordered-event capability |
| contradiction/aggregation/event-ordering coverage providers | BEAM eval profile or delete | Category-shaped compensation |
| scent-trail, broad-raw, quality-fallback and topic-rescue chains | Delete | Repeated post-recall heuristic compensation |
| `_preserve_*`, post-rerank preservation and runtime required preservation | Delete | Product selector is one-pass |
| category-heavy evidence-pack expansion | BEAM eval profile or delete | Model/benchmark shaping, not product recall |
| `_sanitize_model_call`, `_model_call_summary`, `_labeled_precision` | `api/service_telemetry.py` | Non-retrieval service reporting |
| `_explicit_order_mentions` | `ingestion/order_markers.py` | Write-side event-edge parsing |

## Complete Disposition

The entries below are the sole assignment list. There are 46 `MemoryService` methods and 72 `service_helpers` functions, for 118 definitions total.

### Raw/fact/event/view/profile hybrid recall

Destination: Product providers.

- `MemoryService._recall_candidates`
- `MemoryService._retrieval_query`
- `MemoryService._plan_uses_source`
- `MemoryService._source_enabled`

### Entity and lexical providers

Destination: Product entity/lexical providers.

- `MemoryService._entity_candidates`
- `MemoryService._exact_candidates`
- `MemoryService._upsert_span_entities`
- `MemoryService._upsert_fact_entities`
- `MemoryService._upsert_event_entities`
- `service_helpers._exact_overlap_score`
- `service_helpers._exact_query_terms`
- `service_helpers._cjk_exact_match_phrases`
- `service_helpers._matched_query_conditions`

### Persisted chronology selector

Destination: Product chronology provider.

- `MemoryService._event_ordering_graph_selector_candidates`
- `MemoryService._event_ordering_timeline_candidates`
- `MemoryService._event_ordering_episode_recall_candidates`
- `MemoryService._event_ordering_event_candidates`
- `MemoryService._event_ordering_sort_key`
- `MemoryService._event_ordering_observed_at`
- `MemoryService._candidate_timeline_position`
- `MemoryService._coerce_datetime`
- `MemoryService._event_id`
- `MemoryService._resolve_event`
- `MemoryService._event_edge`
- `service_helpers._date_signal`
- `service_helpers._temporal_target_roles_for_service`
- `service_helpers._temporal_roles_in_text`
- `service_helpers._temporal_focus_terms_for_service`
- `service_helpers._candidate_in_timeline_window`
- `service_helpers._natural_turn_key`
- `service_helpers._service_date_scope_labels`

### Contradiction, aggregation, and event-ordering coverage providers

Destination: BEAM eval profile or delete.

- `MemoryService._event_ordering_shadow_coverage`
- `MemoryService._event_ordering_coverage_candidates`
- `MemoryService._contradiction_claim_candidates`
- `MemoryService._temporal_coverage_candidates`
- `MemoryService._aggregation_coverage_candidates`
- `service_helpers._source_coverage`
- `service_helpers._dedupe_event_ordering_support_events`
- `service_helpers._surface_claim_polarity`
- `service_helpers._aggregation_query_terms`
- `service_helpers._aggregation_signal`
- `service_helpers._adjacent_assistant_recommendation_spans`
- `service_helpers._aggregation_recommendation_request_signal`
- `service_helpers._recommendation_request_specificity`
- `service_helpers._assistant_recommendation_list_signal`
- `service_helpers._synthesis_evidence_signal`
- `service_helpers._is_cross_factor_synthesis_query`
- `service_helpers._synthesis_candidate_key`
- `service_helpers._aggregation_focus_priority`
- `service_helpers._is_generic_count_or_list_query`
- `service_helpers._generic_aggregation_keys`
- `service_helpers._clean_generic_aggregation_key`
- `service_helpers._quoted_title_candidates`
- `service_helpers._normalize_title_key`
- `service_helpers._is_non_title_quote`
- `service_helpers._key_diverse_aggregation_candidates`
- `service_helpers._aggregation_scene_representatives`
- `service_helpers._aggregation_context_support_candidate`
- `service_helpers._aggregation_group_support_specificity`
- `service_helpers._high_value_aggregation_context_support`
- `service_helpers._aggregation_query_date_support`
- `service_helpers._aggregation_context_specificity`
- `service_helpers._aggregation_query_context_keys`
- `service_helpers._aggregation_context_features`
- `service_helpers._is_broad_exploration_aggregation_query`

### Scent-trail, broad-raw, quality-fallback, and topic-rescue chains

Destination: Delete.

- `MemoryService._topic_scoped_raw_candidates`
- `MemoryService._raw_scent_trail_candidates`
- `MemoryService._broad_raw_recall_candidates`
- `MemoryService._filter_stale_current_value_candidates`
- `MemoryService._apply_topic_scope_filter`
- `MemoryService._event_ordering_candidate_allowed_in_topic_scope`
- `MemoryService._apply_event_ordering_post_preservation_topic_scope_filter`
- `MemoryService._apply_quality_fallback`
- `MemoryService._quality_fallback_candidates`
- `MemoryService._candidate_in_topic_groups`
- `service_helpers._broad_recall_candidate_allowed`
- `service_helpers._replaceable_low_synthesis_index`
- `service_helpers._topic_scope_group_limit`
- `service_helpers._topic_scope_groups`
- `service_helpers._topic_scope_score`
- `service_helpers._topic_phrase_bonus`
- `service_helpers._topic_anchor_phrases_for_service`
- `service_helpers._clean_topic_anchor_for_service`
- `service_helpers._topic_anchor_score_for_service`
- `service_helpers._topic_scope_tokens`
- `service_helpers._expand_topic_tokens`
- `service_helpers._span_group_key`
- `service_helpers._broad_raw_recall_queries`
- `service_helpers._intent_string_list`
- `service_helpers._intent_recall_signal`
- `service_helpers._preference_recall_terms`
- `service_helpers._preference_recall_signal`
- `service_helpers._scent_trail_queries`
- `service_helpers._ordered_topic_scope_tokens`
- `service_helpers._scent_trail_score`
- `service_helpers._quality_fallback_terms`
- `service_helpers._fallback_salience_score`

### Preservation and post-rerank selection

Destination: Delete.

- `MemoryService._preserve_quota_after_rerank`
- `MemoryService._preserve_high_signal_exact`
- `MemoryService._preserve_temporal_coverage`
- `MemoryService._preserve_contradiction_claim_coverage`
- `MemoryService._preserve_scent_trail`
- `MemoryService._preserve_event_ordering_raw_facets`
- `MemoryService._preserve_broad_raw_recall`
- `MemoryService._preserve_aggregation_coverage`
- `MemoryService._preserve_user_synthesis_anchors`
- `MemoryService._preserve_event_ordering_events`
- `MemoryService._preserve_high_ranked_summaries`

### Category-heavy evidence-pack expansion

Destination: BEAM eval profile or delete.

- `service_helpers._has_value_intent`
- `service_helpers._has_current_intent`
- `service_helpers._compatible_value_mention`
- `service_helpers._value_signal`
- `service_helpers._current_state_signal`
- `service_helpers._code_identifier_signal`

### Service telemetry

Destination: `api/service_telemetry.py`.

- `service_helpers._sanitize_model_call`
- `service_helpers._model_call_summary`
- `service_helpers._labeled_precision`

### Write-side order-marker parsing

Destination: `ingestion/order_markers.py`.

- `service_helpers._explicit_order_mentions`

## Coverage Check

The inventory is intentionally definition-based rather than call-site-based. The source definition lists contain 46 `MemoryService` methods and 72 top-level helper functions. Extract the backticked `MemoryService.*` and `service_helpers.*` entries in the complete disposition section, compare them to those source lists, and check that each extracted identifier occurs once. This verifies both exhaustive coverage and the absence of duplicate destination assignments.
