# src/oceanembed/datasets/registry/

This directory will contain the Python dataset registry for OceanEmbed.

## Status

**Not yet implemented.** The registry will be built after:

1. The acquisition script inventory is completed.
2. Dataset audits are conducted in `oceanembed-research/dataset-audits/`.
3. Dataset registry field definitions are agreed upon.

## Intended purpose

The registry maps dataset identifiers to their metadata and adapter classes.
It will allow the rest of the pipeline to work with datasets by name without
hard-coding source-specific details throughout the codebase.

## Do not invent registry entries

Do not add placeholder registry entries or field definitions at this stage.
Registry structure and values will be derived from audited dataset information.
