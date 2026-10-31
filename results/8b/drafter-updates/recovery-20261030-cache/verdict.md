# T5 domain adaptation verdict

substantial_benefit_in_recorded_conditions

Simulated acceptance was used only for development selection. Serving cells use paired GPU-swapped repeats.

- adapt-mult-n8k-r0: conditional_specialist_tradeoff; forgetting regression: True.
  - multilingual k=7 C=1: worth_considering_for_this_cell; first-repeat throughput 254.1 to 329.4 tok/s, median TPOT 3.55 to 2.38 ms. Both repeats and absolute cost/latency tradeoffs are in verdict.json.
- adapt-summ-n8k-r0: conditional_specialist_tradeoff; forgetting regression: True.
  - summarization k=7 C=1: conditional_tradeoff; first-repeat throughput 195.9 to 239.3 tok/s, median TPOT 4.60 to 3.73 ms. Both repeats and absolute cost/latency tradeoffs are in verdict.json.
- adapt-chat-n8k-r0: do_not_implement_from_this_experiment; forgetting regression: True.

Calibrated correctness is consistent with losslessness, not proof. No production deployment or routing change was made.
Whole-recovery allocated compute: 55.47 GPU-hours. No monetary tariff assumed.
