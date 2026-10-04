| Request | n | `<tool_call>` top-1 | Mean probability |
|---|---:|---:|---:|
| L0: empty user turn | 1 | 0/1 (0.0%) | 0.0005 |
| L1: Hello | 1 | 0/1 (0.0%) | 4.50e-20 |
| L2: unrelated factual question | 1 | 0/1 (0.0%) | 3.43e-17 |
| L3: task body only | 300 | 97/300 (32.3%) | 0.3310 |
| L4: predeclared neutral verb + body | 300 | 260/300 (86.7%) | 0.8547 |
| L5: analysis verb + body | 300 | 0/300 (0.0%) | 0.0080 |
| L6: execution verb + body | 300 | 300/300 (100.0%) | 0.9993 |
| L7: no system scaffold + execution + body | 300 | 0/300 (0.0%) | 6.56e-17 |
