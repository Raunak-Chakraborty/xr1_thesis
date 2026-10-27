# XR1 offline Doosan RGB frame cache

Version: 2026-10-27. Opt-in offline PNG loading.

Files relative to Xiaomi-Robotics-1/xr1/:
- `tools/doosan_offline_frame_cache.py`: exact Doosan training-sampler replay + selective FFmpeg sequential PNG extraction.
- `mibot/data/datasets/doosan_cached_images.py`: fail-closed image loader, selected by DoosanJsonDataset when XR1_DOOSAN_OFFLINE_FRAME_ROOT is set.

This tooling does not change original MP4s, episode JSON, language instructions, model architecture, decoder patch, or checkpoints. Its cache is stored on Robotics SSD outside Git; do not commit generated image data.

The cache builder replicates the October 2026 `DoosanJsonDataset._samples()` selection scheme. If sampler implementation, world size, seed, max steps or batch size change, rebuild or extend the cache. Output manifest explains the exact selection parameters. Run a 4-video pilot first. Do not point a training job to this incomplete pilot cache.

The old Decord reader remains untouched. To route training through offline images, validate index/pixel parity first, then add an explicit opt-in to the Doosan dataset's `_images` method in a separate audited patch. Set XR1_DOOSAN_OFFLINE_FRAME_ROOT in the launcher only after a complete cache has been built.


## Validated training milestone

On October 27, 2026, the XR1-5B Doosan last-1-layer Language V2
experiment completed 2,000/2,000 optimizer steps using the offline
PNG frame cache, SDPA, BF16, and two RTX A5000 GPUs.

The original MP4 files and converted episode JSONs remain unchanged.
Online Decord decoding is retained as a separate opt-out path.

The frame cache is external to Git and must be generated for the
training sampler configuration. Cache misses must not silently
fall back to online decoding.

Language policy: configs/language/doosan_assembly_language_v2.yaml
SHA256: 21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d
