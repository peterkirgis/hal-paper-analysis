#!/bin/bash
# Full run script for all benchmarks - processes both generalist and specialist agents where applicable

# USACO - supports both generalist and specialist
echo "Processing USACO..."
python usaco.py --agent-type generalist --verbose > logs/usaco_generalist.log 2>&1
python usaco.py --agent-type specialist --verbose > logs/usaco_specialist.log 2>&1

# GAIA - supports both generalist and specialist
echo "Processing GAIA..."
python gaia.py --agent-type generalist --verbose > logs/gaia_generalist.log 2>&1
python gaia.py --agent-type specialist --verbose > logs/gaia_specialist.log 2>&1

# SWE-bench Mini - supports both generalist and specialist
echo "Processing SWE-bench Mini..."
python swe_bench_mini.py --agent-type generalist --verbose > logs/swebench_generalist.log 2>&1
python swe_bench_mini.py --agent-type specialist --verbose > logs/swebench_specialist.log 2>&1

# CoreBench - supports both generalist and specialist
echo "Processing CoreBench..."
python core_bench.py --agent-type generalist --verbose > logs/corebench_generalist.log 2>&1
python core_bench.py --agent-type specialist --verbose > logs/corebench_specialist.log 2>&1

# SciCode - supports both generalist and specialist
echo "Processing SciCode..."
python sci_code.py --agent-type generalist --verbose > logs/scicode_generalist.log 2>&1
python sci_code.py --agent-type specialist --verbose > logs/scicode_specialist.log 2>&1

# ScienceAgent - supports both generalist and specialist
echo "Processing ScienceAgent..."
python scienceagent.py --agent-type generalist --verbose > logs/scienceagent_generalist.log 2>&1
python scienceagent.py --agent-type specialist --verbose > logs/scienceagent_specialist.log 2>&1

# TAU-bench - supports both generalist and specialist
echo "Processing TAU-bench..."
python tau_bench.py --agent-type generalist --verbose > logs/taubench_generalist.log 2>&1
python tau_bench.py --agent-type specialist --verbose > logs/taubench_specialist.log 2>&1

# AssistantBench - supports both generalist and specialist
echo "Processing AssistantBench..."
python assistant_bench.py --agent-type generalist --verbose > logs/assistantbench_generalist.log 2>&1
python assistant_bench.py --agent-type specialist --verbose > logs/assistantbench_specialist.log 2>&1

# Online Web2Mind - check if it has agent types
# python online_web2mind.py --verbose > logs/online_web2mind.log 2>&1

echo "All benchmark processing complete!"