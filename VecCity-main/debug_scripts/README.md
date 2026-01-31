# Debug Scripts Archive

This directory contains historical debugging and testing scripts used during the development and fixing of the HRNR_Hyperbolic model. These scripts are preserved for reference but are not needed for normal usage.

## Purpose

These scripts were instrumental in diagnosing and fixing the **spatial collapse issue** in HRNR_Hyperbolic, which resulted in a **5000x improvement** in spatial component preservation.

## Script Categories

### Diagnosis Scripts
Files that helped identify problems:
- `diagnose_spatial_collapse.py` - Initial diagnosis of spatial component collapse
- `diagnose_graph_encoder_collapse.py` - Layer-by-layer tracking through graph encoder
- `diagnose_cd_*.py` - Various diagnostic tools for the cd dataset
- `debug_layer1_detailed.py` - Detailed operation-by-operation tracking
- `verify_origin_collapse.py` - Verification of embeddings clustering at origin
- `check_embeddings.py` - Embedding quality checker

### Test Scripts
Files that validated fixes:
- `test_graph_conv_fix.py` - Tested graph convolution fix
- `test_hyperbolic_update_fix.py` - Tested hyperbolic update fix
- `test_optimized_params.py` - Tested parameter optimizations
- `test_norm_preservation.py` - Tested norm preservation mechanisms
- `test_angle_between_versions.py` - Tested angle calculation versions
- `test_log_map_stability.py` - Tested logarithmic map stability
- `test_with_fresh_init.py` - Tested with fresh initialization
- `test_fix.py` - General fix testing
- `debug_fix.py` - Debug mode testing

### Fix Scripts
Files that experimented with solutions:
- `fix_aggregate_to_cluster.py` - Experimented with aggregation fixes

## Key Findings

The debugging process revealed three critical issues:

1. **Euclidean Aggregation** (99% loss)
   - Diagnosed by: `diagnose_spatial_collapse.py`
   - Fixed in: `HRNR_Hyperbolic.py::_aggregate_to_cluster()`

2. **Graph Convolution Gain** (98% loss)
   - Diagnosed by: `debug_layer1_detailed.py`
   - Fixed in: `hyperbolic_utils.py::HyperbolicGraphConv`

3. **Hyperbolic Update** (77% loss)
   - Diagnosed by: `test_hyperbolic_update_fix.py`
   - Fixed in: `HRNR_Hyperbolic.py::_hyperbolic_update()`

## Usage

These scripts are **not required** for normal usage of HRNR_Hyperbolic. They are preserved as:
- Historical record of the debugging process
- Reference for understanding the fixes
- Potential templates for future debugging

## For Normal Usage

Please refer to the main documentation:
- **Complete Guide**: [../docs/HRNR_HYPERBOLIC_COMPLETE_GUIDE.md](../docs/HRNR_HYPERBOLIC_COMPLETE_GUIDE.md)
- **Quick Start**: See main [README.md](../README.md)

All fixes from these debugging sessions have been integrated into the production code.
