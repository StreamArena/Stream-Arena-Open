# Stream-Arena results

Scores range from 0 to 100; unresolved measurements show their lower bound. Overall is the equal-weight mean of the applicable content metrics; Common-4 averages Quality, A–V Sync, Stability and Dialogue. Timing and coverage diagnostics stay local. Rows are alphabetical; no significance ranking is assigned. \* provisional metric; N/A not applicable. Definitions: [docs/evaluation.md](../docs/evaluation.md).

## Digital-human dialogue · core conditions

| System | Quality | A–V Sync\* | Stability | Dialogue | Identity | Overall | Common-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AptAvatar 14B† | 68.03 | 70.32 | 94.74 | 35.98 | 81.00 | 70.01 | 67.26 |
| Avatar-Forever 22B† | 50.86 | 48.16 | 68.48 | 54.92 | 75.23 | 59.53 | 55.60 |
| AvatarForcing 1.3B† | 57.17 | 0.00 | 87.30 | 8.93 | 66.85 | 44.04 | 38.35 |
| AVTR-1† | 82.97 | 69.37 | 99.87 | 54.97 | 88.54 | 79.14 | 76.79 |
| LiveAvatar FULL‡ | 76.27 | 0.00 | 100.00 | 61.36 | N/A | 59.40 | 59.40 |
| LiveTalk 1.3B† | 61.75 | 62.79 | 86.51 | 54.40 | 77.51 | 68.59 | 66.36 |
| MuseTalk 1.5† | 95.34 | 38.09 | 97.11 | 54.34 | 88.44 | 74.66 | 71.22 |
| Quark Live Avatar 14B† | 56.00 | 64.91 | 70.90 | 55.01 | 78.10 | 64.98 | 61.70 |
| Runway Characters | 57.81 | 49.82 | 75.94 | 43.53 | 60.40 | 57.49 | 56.77 |
| SoulX-FlashHead (Lite)† | 68.43 | 29.76 | 91.61 | 56.65 | 79.69 | 65.23 | 61.61 |
| SoulX-FlashTalk 14B† | 68.90 | 66.26 | 95.45 | 50.57 | 83.31 | 72.90 | 70.29 |
| SoulX-LiveAct 18B† | 64.93 | 50.29 | 91.54 | 54.98 | 75.09 | 67.36 | 65.43 |
| Tavus CVI Full‡ | 75.04 | 94.44 | 99.95 | 73.91 | N/A | 85.83 | 85.83 |
| Vidu S2 Avatar | 96.78 | 84.87 | 97.83 | 60.57 | 90.12 | 86.03 | 85.01 |
| Vivix A1 | 45.78 | 69.15 | 66.83 | 50.38 | 75.19 | 61.46 | 58.03 |

† Renderer with the shared ASR, dialogue and TTS stack. ‡ Provider-supplied avatar; Identity is N/A and Overall omits it.

## Avatar performance · core conditions

| System | Quality | A–V Sync\* | Stability | Following | Identity | Overall |
|---|---:|---:|---:|---:|---:|---:|
| AptAvatar 14B | 55.55 | 98.89 | 78.06 | 13.03 | 81.33 | 65.37 |
| Avatar-Forever 22B | 34.73 | 34.66 | 45.31 | 6.37 | 74.82 | 39.17 |
| AvatarForcing 1.3B | 54.32 | 18.44 | 71.60 | 8.44 | 68.50 | 44.26 |
| LiveTalk 1.3B | 71.27 | 86.44 | 96.06 | 16.59 | 83.89 | 70.85 |
| Quark Live Avatar 14B | 43.51 | 98.89 | 55.30 | 11.70 | 79.75 | 57.82 |
| SoulX-LiveAct 18B | 48.39 | 48.67 | 64.94 | 10.74 | 69.83 | 48.51 |
| Vidu S2 Avatar§ | 63.68 | 99.60 | 97.25 | 16.37 | 86.12 | 72.60 |

§ Text-derived line and action inputs; the other systems receive supplied audio.

## World generation · core conditions

| System | Quality | Stability | Following | Continuity | Overall |
|---|---:|---:|---:|---:|---:|
| Causal Forcing++ | 16.02 | 15.83 | 3.70 | 9.04 | 11.14 |
| Context Forcing | 35.70 | 31.03 | 7.77 | 16.49 | 22.75 |
| Helios (Distilled) | 28.79 | 28.03 | 4.44 | 19.89 | 20.29 |
| Krea Realtime | 35.61 | 42.79 | 9.62 | 22.12 | 27.53 |
| LongLive 2.0 | 30.08 | 31.84 | 9.25 | 20.74 | 22.98 |
| MemFlow | 40.93 | 40.38 | 9.25 | 29.54 | 30.02 |
| Reward Forcing | 21.72 | 16.05 | 4.07 | 21.40 | 15.81 |
| Rolling Forcing | 45.19 | 44.47 | 6.66 | 24.61 | 30.23 |
| Self-Forcing | 16.28 | 17.05 | 5.92 | 13.62 | 13.22 |
| ShotStream | 33.80 | 20.54 | 6.66 | 9.34 | 17.58 |
| Yume 1.5 (720p) | 25.96 | 27.67 | 5.18 | 26.22 | 21.26 |
| Zing-0.5 | 35.60 | 39.14 | 2.96 | 23.65 | 25.34 |

## Instruction editing · core conditions

| System | Quality | Stability | Following | Edit Success\* | Preservation | Overall |
|---|---:|---:|---:|---:|---:|---:|
| EditaLive (14B) | 7.86 | 4.12 | 12.03 | 4.26 | 23.15 | 10.28 |
| JoyAI-Video-Edit | 11.62 | 9.57 | 63.89 | 43.52 | 83.33 | 42.39 |
| Krea Realtime 14B V2V | 1.13 | 0.06 | 0.00 | 0.00 | 0.00 | 0.23 |
| LiveEdit (1.3B) | 3.22 | 0.62 | 5.37 | 1.67 | 7.04 | 3.58 |
| SANA-Streaming 2B | 15.15 | 9.22 | 11.48 | 4.63 | 44.26 | 16.94 |
| StreamDiffusion | 0.04 | 0.00 | 0.00 | 0.00 | 1.30 | 0.26 |
| StreamDiffusionV2 1.3B | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| StreamV2V | 0.00 | 0.00 | 0.00 | 0.00 | 0.56 | 0.11 |
| Vidu S2 Editing§ | 22.98 | 13.93 | 72.10 | 57.20 | 60.93 | 45.43 |
| XMax X2.0 | 5.14 | 0.97 | 11.30 | 2.59 | 28.52 | 9.70 |

§ Instruction-to-reference composition.

## Reference editing · core conditions

| System | Quality | Stability | Reference | Edit Success\* | Preservation | Overall |
|---|---:|---:|---:|---:|---:|---:|
| JoyAI-Video-Edit | 6.54 | 3.46 | 26.11 | 8.89 | 26.67 | 14.33 |
| Scope (LongLive 1.3B + VACE) | 2.11 | 2.19 | 0.00 | 0.00 | 2.78 | 1.42 |
| Scope (StreamDiffusionV2 + VACE) | 0.00 | 0.00 | 0.28 | 0.00 | 0.00 | 0.05 |
| Vidu S2 Editing | 37.71 | 46.74 | 19.72 | 15.00 | 64.44 | 36.72 |
| XMax X2.0 | 8.20 | 7.51 | 20.56 | 1.11 | 10.00 | 9.47 |
