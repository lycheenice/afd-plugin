## Background

P2pNcclAFDConnector (implemented with vLLM's PyNcclCommunicator) is a GPU-backed point-to-point connector for CUDA deployments. This guide documents its usage, configuration contract, rank mapping, and operational limits to help users and maintainers configure it correctly.

## When to use `P2pNcclAFDConnector`

Use this connector for CUDA deployments that disaggregate Attention and FFN workers and exchange hidden states through NCCL point-to-point communication. The default path is synchronous. An experimental eager-only multi-stream path is available for the narrow `1A1F`/two-ubatch scope documented below.

It supports both prefill and decode which all support eager mode. CUDA graph support is currently limited to `FULL_DECODE_ONLY`, which is mainly used in decode instance. The checked-in DeepSeek V2 Lite recipes cover colocated and prefill/decode-disaggregated deployments.

For a code-level analysis of the current two-ubatch execution order, the
difference between NCCL asynchronous submission and compute/communication
overlap, and the reasons DBO may show no performance gain, see
[NV GPU DBO execution and overlap analysis](NV_GPU_DBO_EXECUTION_ANALYSIS.md).


## How it works

Throughout this section, let `A = num_attention_ranks` and `F = num_ffn_ranks`. One side must be an integer multiple of the other. `ratio = max(A, F) / min(A, F)` is therefore a whole number. One physical process sits at up to three different rank numbers — an AFD world rank, a subgroup rank, and a control-plane (`p2p`) rank — all derived deterministically from the role, role rank, and topology counts.

### AFD world (shared rendezvous)

The connector creates one AFD NCCL world of size `F + A`, rendezvoused at `tcp://host:port`, ordered FFN-first:

```text
world rank:   0    1   ...  F-1   F    F+1  ...  F+A-1
member:       F0   F1  ...  F_    A0   A1   ...  A_
```

FFN role rank `i` gets world rank `i`; Attention role rank `j` gets world rank `F + j`. This world fixes the global ordering; the actual traffic flows through the two derived structures below.

### Data plane: one subgroup per FFN rank

For `A >= F`, each FFN rank `k` owns subgroup `k`, containing itself plus its `ratio` consecutive Attention peers `A(k*ratio) .. A(k*ratio + ratio - 1)`. The FFN rank is subgroup rank `0` and the Attention peers occupy ranks `1..ratio`. For `F > A`, the layout is reversed: each Attention rank owns a subgroup at rank `0` with its consecutive FFN peers at ranks `1..ratio`; Attention activations are replicated to those peers and the designated peer returns the result.

Each subgroup is its own process group rendezvoused on `port + subgroup_index + 1` (this is where the derived-port requirement comes from), carrying two NCCL communicators: Attention-to-FFN for hidden states and FFN-to-Attention for FFN outputs. Per layer/stage, each Attention peer sends its hidden states to subgroup rank `0`; the FFN rank receives from ranks `1..ratio` in order, concatenates along the token dimension, runs FFN work, splits the output by the recorded sequence lengths, and sends each slice back to the originating Attention rank. The data path uses vLLM `PyNcclCommunicator.send()` / `recv()` on the current CUDA stream.

### Control plane: the DP metadata group

Tensor shapes vary per step (token counts per DP rank), so before data transfers can be posted, the FFN side needs the per-stage token counts. A separate NCCL process group named `p2p`, of size `F + min_size = 2F`, carries this DP metadata. It rendezvouses at the same `tcp://host:port` as the AFD world but as a distinct group:

- All `F` FFN ranks join, keeping their role rank as their `p2p` rank (`0..F-1`).
- Only the first `min_size` Attention ranks (role ranks `0..F-1`) join, at `p2p` ranks `F..2F-1`. Attention ranks with role rank `>= F` do not participate; their metadata send is a no-op.

The pairing is 1:1 by index: Attention role rank `r` sends the control payload to FFN rank `r`. The payload carries the whole per-rank token-count vector plus graph-capture/warmup flags, so one sender per FFN rank is enough — each FFN rank derives every Attention peer's token count from that vector, including peers that are not in the control-plane group.

In code, the control plane is a pluggable module rather than part of the connector interface. `P2pNcclAFDConnector` creates a `P2pNcclAFDControlPlane` instance — an implementation of the `AFDControlPlane` contract from `afd_plugin/connectors/base.py` — and exposes it as `connector.control_plane`. The Attention-side runner calls `control_plane.update_state_from_dp_metadata(...)` and `control_plane.send_dp_metadata_list(...)` before each step; the FFN worker loop blocks on `control_plane.recv_dp_metadata_list()`. Because the connector exposes a `control_plane`, each FFN step is triggered by the arrival of a DP metadata payload (connectors without a DP metadata control plane leave `control_plane` as `None` and drive FFN steps from the connector itself).

### Worked example: `4A2F` (`ratio = 2`, `min_size = 2`)

```text
AFD world (port):        F0=0  F1=1  A0=2  A1=3  A2=4  A3=5

Data plane:
  subgroup 0 (port+1):   F0(rank 0) <-> A0(rank 1), A1(rank 2)
  subgroup 1 (port+2):   F1(rank 0) <-> A2(rank 1), A3(rank 2)

Control plane "p2p" group (port, size 4):
  p2p ranks:             F0=0  F1=1  A0=2  A1=3      (A2, A3 excluded)
  metadata flow:         A0 -> F0,   A1 -> F1
```

All three groups are derived from just `host`, `port`, the role, and the rank counts; any mismatch across processes makes the collective rendezvous hang or fail. Note that `num_attention_ranks` and `num_ffn_ranks` are worker totals, not engine counts: attention DP=2 x TP=2 means `num_attention_ranks=4`, and the connector expands the per-DP token vector across TP peers when the vector is shorter than the attention rank count.

## Configuration

AFD configuration is supplied through vLLM's `--additional-config` under the `afd` key. The presence of the `afd` object enables AFD; omit it to disable AFD. Attention and FFN processes must use the same rendezvous address and topology counts; only `role` and the device assignment differ. The plugin selects the role-specific worker automatically.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 1,
    "num_ffn_ranks": 1,
    "afd_role_rank": 0,
    "compute_gate_on_attention": false,
    "connector_extra_config": {}
  }
}
```

### Fields

| Field | Type | Default | Required / meaning |
| --- | --- | --- | --- |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. Attention sends hidden states; FFN receives and returns FFN outputs. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Must be `P2pNcclAFDConnector` for this GPU path. |
| `host` | `str` | `"127.0.0.1"` | Non-empty rendezvous/control-plane host. All participating ranks must use a reachable, identical value. Host must be the first rank of FFN.|
| `port` | `int` | `1239` | Base rendezvous port, valid range `1..65535`. The connector also uses `port + subgroup_index + 1`, so those ports must be free and reachable. |
| `num_attention_ranks` | `int` | `1` | Total number of AFD Attention ranks, including DP/TP-derived worker ranks. Must be positive. |
| `num_ffn_ranks` | `int` | `1` | Total number of AFD FFN ranks, including DP/TP-derived worker ranks. Must be positive. |
| `afd_role_rank` | `int` | `0` | Rank within the selected role group. Must satisfy `0 <= rank < num_<role>_ranks`. Runners normally derive it from DP/PCP/TP placement; users should not assign duplicate role ranks. |
| `compute_gate_on_attention` | `bool` | `false` | Must be `false`. Whether Attention computes MoE gate outputs before sending work to FFN. This is a general AFD field, not a PyNccl transport setting. |
| `connector_extra_config` | `dict` | `{}` | Connector-owned options. Supports `async_transfer` and `async_slots` as described below; unknown fields are rejected. |
| `async` / `async_dp` | `bool` | `false` | Must remain `false` for `P2pNcclAFDConnector`; AFD async mode requires `CAMAsyncAFDConnector`. |

Compatibility aliases currently accepted are `afd_role`, `afd_connector`, `afd_host`, and `afd_port`. Canonical field names should be used in new examples.

### Experimental eager multi-stream path

This is separate from the general AFD `async` / `async_dp` mode. Enable it identically on the Attention and FFN processes:

```jsonc
"connector_extra_config": {
  "async_transfer": true,
  "async_slots": 2
}
```

The current MVP fails fast unless all of the following hold:

- eager execution (`--enforce-eager`);
- `1A1F` topology;
- DBO enabled with exactly two ubatches;
- `async_slots` equals `2`.

It uses connector-owned A2F and F2A CUDA streams, CUDA events for compute/transfer dependencies, and bounded receive buffers. The synchronous path remains the default and CUDA graph is not supported by this experimental path. gpu-host DeepSeek-V2-Lite validation has established semantic equivalence to the synchronous reference and profiler-visible compute/communication overlap. Performance, concurrent soak, failure-path, and broader-topology validation remain incomplete, so this mode is still experimental and is not recommended for production use. See `experiment/DSV2_LITE_GPU_ASYNC_DESIGN.md` for the evidence and remaining gates.

## Topology rules

`P2pNcclAFDConnector` currently requires one side to be an integer multiple of the other:

```text
max(num_attention_ranks, num_ffn_ranks) % min(num_attention_ranks, num_ffn_ranks) == 0
```

Therefore, each rank on the smaller side maps to the same integer number of consecutive peers on the larger side.

Examples:

| Topology | Valid? | Mapping |
| --- | --- | --- |
| `1A1F` | Yes | `F0 <-> A0` |
| `2A2F` | Yes | `F0 <-> A0`, `F1 <-> A1` |
| `4A2F` | Yes | `F0 <-> A0,A1`, `F1 <-> A2,A3` |
| `1A2F` | Yes | `A0 <-> F0,F1`; Attention activations fan out to both FFN peers. |
| `2A4F` | Yes | `A0 <-> F0,F1`, `A1 <-> F2,F3` |
| `3A2F` | No | Attention rank count is not divisible by FFN rank count. |

## Minimal launch shape

Attention side:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

FFN side:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/model \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The HTTP ports (`18000` / `18001` above) are vLLM service ports and are separate from the AFD NCCL rendezvous base port (`6239`). Use distinct CUDA devices for the two roles.

For complete `1A1F`, `2A2F`, `4A4F`, eager, DBO, and CUDA graph examples, see `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite`.

## Requirements and limitations to document in code

- CUDA-capable PyTorch/vLLM environment with NCCL and vLLM's `PyNcclCommunicator` available.
- Hidden-state tensors must reside on CUDA devices; CPU tensors are rejected.
- All ranks must agree on `host`, `port`, rank counts, model hidden size/dtype, and role-rank assignment.
- The rendezvous base port and derived subgroup ports must be free and reachable.
- Initialization is collective: missing ranks, mismatched counts, or duplicate role ranks can cause initialization failure or timeout.
- Current GPU CUDA graph support is `FULL_DECODE_ONLY`; GPU DBO plus CUDA graph is limited to exactly two ubatches.
- To enable DBO, set `--enable-dbo`, and configure the threshold with `--dbo-decode-token-threshold` and `--dbo-prefill-token-threshold`. See `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite` for examples.
- The repository recipes currently validate specific GPU layouts (including DeepSeek V2 Lite and tested A/H-class hardware). Cross-node use depends on NCCL/network configuration and is not established by the current recipes; document it as unverified rather than promising transparent fallback.
- There is no automatic fallback from this connector to another transport. Select an NPU connector explicitly on Ascend.
