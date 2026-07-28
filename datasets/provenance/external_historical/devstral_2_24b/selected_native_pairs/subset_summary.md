# Devstral Clean-400 Subset

- Source rows: `500`
- Eligible rows: `309`
- Selected rows: `309`
- Criterion: `clean top1 = [TOOL_CALLS]` and `corrupt top1 != [TOOL_CALLS]` on rescored real prompts.
- Ranking: descending `clean_tool_prob - corrupt_tool_prob`.
