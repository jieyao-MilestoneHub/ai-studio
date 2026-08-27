# twin

A framework for building a personal "digital twin" agent: feed in one
person's (the principal's) historical data and self-report interviews, get
back an agent whose judgment, tool-use style, and proactivity — including
the tendency *not* to act — resembles that person in situations they've
never actually faced.

**Status**: early scaffold. The system is fully specified in
[`reference/SPEC.md`](reference/SPEC.md) (with acceptance criteria in
[`reference/EVAL.md`](reference/EVAL.md) and the onboarding interview
protocol in [`reference/INTERVIEW.md`](reference/INTERVIEW.md)) and the
build is tracked phase by phase in [`PLAN.md`](PLAN.md). See
[`CLAUDE.md`](CLAUDE.md) for the architecture and the non-negotiables.

## Third-party content

This project ingests one person's personal data, which routinely includes
messages, photos, and other content involving other people ("third
parties") who have not consented to being modeled. **You, the user running
this project, are solely responsible for the legality of any third-party
content in your own data** — this project does not anonymize, de-identify,
or otherwise handle that on your behalf (`SPEC.md` §8). It only provides
technical guardrails (`third_party_spans` tagging at ingest time, plus a
`.gitignore` and pre-commit hook that hard-block `data/`, `adapters/`,
`transcripts/`, and `eval/` from version control) so that whatever policy
you choose is enforceable and reversible, not so that you don't need one.
