# Granite 500-pair dataset

- n_pairs: 500
- clean tool-call rate: 1.0000
- corrupt tool-call rate: 0.0000
- clean minus corrupt gap: 1.0000
- language_counts: {'python': 57, 'cpp': 170, 'java': 273}
- clean_candidate_counts: {'save': 167, 'write': 166, 'add': 167}
- corrupt_candidate_counts: {'inspect': 100, 'discuss': 100, 'explore': 100, 'study': 100, 'review': 100}
- quadrant_counts: {'clean_only': 500}

## Selection matrix

| clean | corrupt | selected |
| --- | --- | ---: |
| add | discuss | 0 |
| add | explore | 0 |
| add | inspect | 45 |
| add | review | 61 |
| add | study | 61 |
| save | discuss | 13 |
| save | explore | 31 |
| save | inspect | 49 |
| save | review | 39 |
| save | study | 35 |
| write | discuss | 87 |
| write | explore | 69 |
| write | inspect | 6 |
| write | review | 0 |
| write | study | 4 |
