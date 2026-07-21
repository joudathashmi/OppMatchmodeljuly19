# National-scale launch roadmap

From validated analyst tool to national investment-matching platform.

Current state: a rigorously tested, self-verifying, human-in-the-loop matching
engine, validated at pilot scale (61 companies x 12 opportunities) on a single
machine, with company data flowing to the public OpenAI API. Ready for internal
analyst use today. The gap to a national platform is a program of work, not a
rewrite: the engine, its verification harness, its drift instrumentation and
its human-feedback loop are the parts national platforms usually lack and this
one already has.

Guiding principle: launch the WORKFLOW nationally, not raw model outputs. The
platform recommends; accountable analysts decide; nothing model-generated about
a named company becomes externally visible without human sign-off.

Estimated overall timeline: 6 to 9 months to a governed national launch, with
phases overlapping.

## Phase 0 (weeks 1-4): internal pilot and hygiene

- Put the tool in daily analyst use exactly as it is; every Agree / Not-a-fit
  verdict feeds calibration. Target: 100+ human verdicts (currently 8).
- Replace the borrowed API key with project-owned credentials in a managed
  secret store; rotate keys.
- Write the model card and a decision-governance one-pager: what the tool may
  claim, what requires human sign-off, who owns overrides.
- Exit criteria: analysts use it weekly; verdict pool over 100; secrets clean.

## Phase 1 (months 1-3): shadow validation - prove it against reality

- Run the engine alongside real BD work for a full quarter. Measure:
  - Precision of the pursue list: share of engine recommendations analysts
    actually pursue after review.
  - Analyst time saved per screening cycle.
  - Downstream signal: meetings taken, LOIs or MoUs traceable to engine leads.
- Recalibrate on the grown label pool; freeze a validated baseline (rubric
  hash + weights + thresholds) and record its measured precision.
- Exit criteria: a written validation report, for example "at least 70 percent
  of pursue-list rows accepted by analysts; zero critical false positives
  published". No national claim should precede this evidence.

## Phase 2 (months 1-3, parallel): data foundation

The dataset, not the model, is the binding constraint at national scale.

- Company data pipeline: continuous ingestion from authoritative sources
  (commercial registry data, export databases, chambers, curated vendor data)
  instead of a static spreadsheet. Thousands of companies, refreshed on a
  schedule, with per-record provenance and vintage stamps.
- Entity resolution at real-world messiness: upgrade the canonical-name
  approach to proper ER (identifiers where available, fuzzy matching with
  human adjudication queues).
- Fact-verification layer: any claim that can appear in an externally visible
  assessment (certifications, facilities, footprint) must be checked against a
  source, or labeled as unverified self-description.
- Opportunity intake standardized on the structured questionnaire format the
  readiness gate already expects.
- Anchor-prospect sourcing per vertical (the summary page's standing
  recommendation): build target universes per opportunity class rather than
  matching only against the supplier registry.

## Phase 3 (months 2-4): sovereignty and compliance

- All inference in-tenant: Azure OpenAI deployments (chat AND embeddings) in a
  qualifying region; zero calls to the public API. The code already supports
  this (--chat-provider azure); the embeddings deployment is the missing piece.
- PDPL compliance review (SDAIA): data classification, records of processing,
  retention rules for company data and model outputs.
- Security: NCA Essential Cybersecurity Controls alignment, CST cloud
  framework, penetration test, full audit logging of every model output and
  every human override.
- Align with national AI guidance (SDAIA generative AI guidelines): model
  documentation, human oversight, explainability - most artifacts already
  exist in docs/ and need packaging, not invention.

## Phase 4 (months 3-6): platform engineering

- From script to service: API layer and job queue for matching runs; a
  database replacing CSV and file caches (the schemas are already clean);
  multi-user access with SSO and role-based permissions (analyst, reviewer,
  admin).
- Observability: dashboards for verdict drift (already instrumented per run),
  calibration AUC, cost, latency, and coverage; alerting on anomalies.
- Reproducibility: pinned model versions, run manifests (rubric hash exists),
  environment as code, backup and disaster recovery.
- verify_v3.py becomes a CI gate: no output ships unless all verification
  layers pass.

## Phase 5 (months 5-7): governed beta

- Two or three partner teams inside the organization use the platform for
  live work under the governance rules.
- Correction and appeal workflow for any company named in an assessment;
  human review SLA before anything leaves the building.
- Legal review of externally visible statement templates; standard
  disclaimers; bias and fairness assessment across sector, size and
  geography.
- Exit criteria: beta teams sign off; governance board approves; validation
  report from Phase 1 updated with beta numbers.

## Phase 6: national launch

- Public or cross-government surface only after: outcome-validation evidence,
  security certification, governance sign-off.
- Human-in-the-loop remains mandatory for all published assessments.
- Quarterly recalibration and drift review as standing operations.

## What can honestly be claimed at each stage

- Today: "An AI decision-support engine for investment matching, independently
  verified, human-in-the-loop, validated at pilot scale."
- After Phase 1: "...with measured precision against real analyst decisions."
- After Phase 3: "...running fully in-tenant under national data regulations."
- After Phase 5: "...operating under a governed workflow with correction and
  appeal processes."
- Only then: a national platform.
