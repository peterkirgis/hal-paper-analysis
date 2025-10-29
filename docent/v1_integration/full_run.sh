#!/bin/bash
# Full run script for all benchmarks - processes both generalist and specialist agents where applicable

# USACO - supports both generalist and specialist
echo "Processing USACO..."
python usaco.py --agent-type generalist --verbose --log-dir logs/usaco/generalist
python usaco.py --agent-type specialist --verbose --log-dir logs/usaco/specialist

# GAIA - supports both generalist and specialist
echo "Processing GAIA..."
python gaia.py --agent-type generalist --verbose --log-dir logs/gaia/generalist
python gaia.py --agent-type specialist --verbose --log-dir logs/gaia/specialist

# SWE-bench Mini - supports both generalist and specialist
echo "Processing SWE-bench Mini..."
python swe_bench_mini.py --agent-type generalist --verbose --log-dir logs/swe_bench_mini/generalist
python swe_bench_mini.py --agent-type specialist --verbose --log-dir logs/swe_bench_mini/specialist

# CoreBench - supports both generalist and specialist
echo "Processing CoreBench..."
python core_bench.py --agent-type generalist --verbose --log-dir logs/core_bench/generalist
python core_bench.py --agent-type specialist --verbose --log-dir logs/core_bench/specialist

# SciCode - supports both generalist and specialist
echo "Processing SciCode..."
python sci_code.py --agent-type generalist --verbose --log-dir logs/sci_code/generalist
python sci_code.py --agent-type specialist --verbose --log-dir logs/sci_code/specialist

# ScienceAgent - supports both generalist and specialist
echo "Processing ScienceAgent..."
python scienceagent.py --agent-type generalist --verbose --log-dir logs/scienceagent/generalist
python scienceagent.py --agent-type specialist --verbose --log-dir logs/scienceagent/specialist

# TAU-bench - supports both generalist and specialist
echo "Processing TAU-bench..."
python tau_bench.py --agent-type generalist --verbose --log-dir logs/tau_bench/generalist
python tau_bench.py --agent-type specialist --verbose --log-dir logs/tau_bench/specialist

# AssistantBench - supports both generalist and specialist
echo "Processing AssistantBench..."
python assistant_bench.py --agent-type generalist --verbose --log-dir logs/assistant_bench/generalist
python assistant_bench.py --agent-type specialist --verbose --log-dir logs/assistant_bench/specialist

# Online Web2Mind - check if it has agent types
# python online_web2mind.py --verbose --log-dir logs/online_web2mind

echo "All benchmark processing complete!"