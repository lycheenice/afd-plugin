from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig, afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.distributed import build_rank_mapping  # noqa: E402


def _fake_vllm_config(
    *,
    data_parallel_size=1,
    data_parallel_rank=0,
    enforce_eager=True,
    use_ubatching=False,
    num_ubatches=1,
    extra_config=None,
):
    return SimpleNamespace(
        additional_config={
            "afd": {"connector_extra_config": extra_config or {}},
        },
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            use_ubatching=use_ubatching,
            num_ubatches=num_ubatches,
        ),
    )


def _tolist(value):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def test_p2p_connector_is_registered():
    sys.modules.pop("afd_plugin.connectors.gpu.p2p", None)

    cls = AFDConnectorFactory.get_connector_class("P2pNcclAFDConnector")

    assert cls.__name__ == "P2pNcclAFDConnector"


def test_p2p_connector_can_be_constructed_without_runtime_initialization():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )

    assert connector.is_initialized is False
    assert connector.world_rank == 1
    assert connector.dst_list == [0]


def test_p2p_connector_uses_config_role_rank_not_dp_rank():
    # The runners fold dp/pcp/tp offsets into afd_role_rank before creating
    # the connector; the connector must not re-derive it from the DP rank,
    # otherwise TP peers within one DP group collapse onto the same role rank
    # (e.g. dp2tp2: EP ranks 0..3 would become 0,0,1,1 and collide on the
    # subgroup rendezvous port).
    connector = AFDConnectorFactory.create_connector(
        3,
        3,
        SimpleNamespace(
            additional_config={},
            model_config=SimpleNamespace(
                dtype="bf16",
                enforce_eager=True,
                hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_rank=1,
            ),
        ),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=4,
            num_ffn_ranks=4,
            afd_role_rank=3,
        ),
    )

    assert connector.mapping.role_rank == 3
    assert connector.world_rank == 7
    assert connector.p2p_rank == 7


@pytest.mark.parametrize(
    ("attention_size", "ffn_size", "role", "role_rank", "subgroup_ranks", "dsts"),
    [
        (2, 2, "attention", 1, (1, 3), (1,)),
        (2, 1, "attention", 0, (0, 1, 2), (0,)),
        (4, 2, "attention", 2, (1, 4, 5), ()),
        (4, 2, "ffn", 1, (1, 4, 5), ()),
    ],
)
def test_p2p_topology_supports_equal_and_integer_multiple_attention_counts(
    attention_size,
    ffn_size,
    role,
    role_rank,
    subgroup_ranks,
    dsts,
):
    mapping = build_rank_mapping(
        AFDConfig(
            role=role,
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
            afd_role_rank=role_rank,
        ),
    )

    assert mapping.ratio == attention_size // ffn_size
    assert mapping.subgroup_ranks == subgroup_ranks
    assert mapping.dp_metadata_destinations == dsts


@pytest.mark.parametrize(
    (
        "attention_size",
        "ffn_size",
        "role",
        "role_rank",
        "subgroup_ranks",
        "rank_in_subgroup",
        "dsts",
    ),
    [
        (1, 2, "attention", 0, (2, 0, 1), 0, (0, 1)),
        (1, 2, "ffn", 0, (2, 0, 1), 1, ()),
        (1, 2, "ffn", 1, (2, 0, 1), 2, ()),
        (2, 4, "attention", 1, (5, 2, 3), 0, (2, 3)),
        (2, 4, "ffn", 3, (5, 2, 3), 2, ()),
    ],
)
def test_p2p_topology_supports_integer_multiple_ffn_counts(
    attention_size,
    ffn_size,
    role,
    role_rank,
    subgroup_ranks,
    rank_in_subgroup,
    dsts,
):
    mapping = build_rank_mapping(
        AFDConfig(
            role=role,
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
            afd_role_rank=role_rank,
        ),
    )

    assert mapping.reversed is True
    assert mapping.ratio == ffn_size // attention_size
    assert mapping.subgroup_ranks == subgroup_ranks
    assert mapping.rank_in_subgroup == rank_in_subgroup
    assert mapping.dp_metadata_destinations == dsts


@pytest.mark.parametrize(
    (
        "attention_size",
        "ffn_size",
        "ffn_rank",
        "token_counts",
        "expected_peer_tokens",
    ),
    [
        (2, 1, 0, [3, 5], [3, 5]),
        (4, 2, 0, [3, 5, 7, 11], [3, 5]),
        (4, 2, 1, [3, 5, 7, 0], [7, 1]),
        (6, 3, 2, [2, 3, 5, 7, 11, 13], [11, 13]),
    ],
)
def test_p2p_ffn_metadata_tracks_each_attention_peer_in_xayf(
    attention_size,
    ffn_size,
    ffn_rank,
    token_counts,
    expected_peer_tokens,
):
    connector = AFDConnectorFactory.create_connector(
        ffn_rank,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
            afd_role_rank=ffn_rank,
        ),
    )

    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata(token_counts)},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )

    for src_rank, expected_tokens in enumerate(expected_peer_tokens, start=1):
        assert connector._recv_attn_tensor_metadata_list[
            (0, src_rank)
        ].size == torch.Size([expected_tokens, 16])

    assert connector.tensor_metadata_list[0].size == torch.Size(
        [sum(expected_peer_tokens), 16],
    )


def test_p2p_tensor_metadata_clamps_idle_attention_rank_to_dummy_token():
    ffn_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([0, 4])},
        is_graph_capturing=False,
        is_warmup=False,
    )

    ffn_connector.control_plane.update_state_from_dp_metadata(payload)

    assert ffn_connector._recv_attn_tensor_metadata_list[(0, 1)].size == torch.Size(
        [1, 16],
    )
    assert ffn_connector._recv_attn_tensor_metadata_list[(0, 2)].size == torch.Size(
        [4, 16],
    )
    assert ffn_connector.tensor_metadata_list[0].size == torch.Size([5, 16])

    attention_connector = AFDConnectorFactory.create_connector(
        1,
        1,
        _fake_vllm_config(data_parallel_size=2, data_parallel_rank=0),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    attention_connector.control_plane.update_state_from_dp_metadata(payload)

    assert attention_connector.tensor_metadata_list[0].size == torch.Size([1, 16])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 3,
            },
            "num_ffn_ranks to be a multiple of num_attention_ranks",
        ),
        (
            {
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": 3,
                "num_ffn_ranks": 2,
            },
            "multiple of num_ffn_ranks",
        ),
    ],
)
def test_p2p_topology_validation_errors_are_clear(raw, message):
    with pytest.raises(ValueError, match=message):
        afd_config_from_mapping(raw)


def test_p2p_module_exports_connector_class():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    assert module.P2pNcclAFDConnector.__module__ == "afd_plugin.connectors.gpu.p2p"


def test_p2p_async_extra_config_is_typed_and_defaults_off():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    assert module.P2pNcclExtraInfo.from_mapping(None).to_mapping() == {
        "async_transfer": False,
        "async_slots": 2,
    }
    assert module.P2pNcclExtraInfo.from_mapping(
        {"async_transfer": "true", "async_slots": "2"},
    ).to_mapping() == {"async_transfer": True, "async_slots": 2}


@pytest.mark.parametrize(
    ("raw", "error", "message"),
    [
        ({"unexpected": True}, ValueError, "unknown P2P"),
        ({"async_transfer": "sometimes"}, TypeError, "must be a boolean"),
        ({"async_slots": 0}, ValueError, "must be positive"),
    ],
)
def test_p2p_async_extra_config_rejects_invalid_values(raw, error, message):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    with pytest.raises(error, match=message):
        module.P2pNcclExtraInfo.from_mapping(raw)


@pytest.mark.parametrize(
    (
        "enforce_eager",
        "attention_size",
        "ffn_size",
        "use_ubatching",
        "num_ubatches",
        "async_slots",
        "message",
    ),
    [
        (False, 1, 1, True, 2, 2, "requires eager mode"),
        (True, 2, 1, True, 2, 2, "requires 1A1F topology"),
        (True, 1, 1, False, 2, 2, "exactly two DBO ubatches"),
        (True, 1, 1, True, 1, 2, "exactly two DBO ubatches"),
        (True, 1, 1, True, 2, 3, "requires async_slots=2"),
    ],
)
def test_p2p_async_constructor_rejects_unsupported_scope(
    enforce_eager,
    attention_size,
    ffn_size,
    use_ubatching,
    num_ubatches,
    async_slots,
    message,
):
    with pytest.raises(ValueError, match=message):
        AFDConnectorFactory.create_connector(
            0,
            0,
            _fake_vllm_config(
                enforce_eager=enforce_eager,
                use_ubatching=use_ubatching,
                num_ubatches=num_ubatches,
                extra_config={
                    "async_transfer": True,
                    "async_slots": async_slots,
                },
            ),
            AFDConfig(
                role="attention",
                connector="P2pNcclAFDConnector",
                num_attention_ranks=attention_size,
                num_ffn_ranks=ffn_size,
            ),
        )


def test_p2p_dp_metadata_serialization_uses_json_payload():
    module = importlib.import_module("afd_plugin.connectors.metadata")
    metadata = AFDDPMetadata(num_tokens_across_dp_cpu=[3, 5])

    payload = module.encode_control_payload(
        AFDControlPayload(
            dp_metadata_list={7: metadata},
            is_graph_capturing=True,
            is_warmup=False,
        ),
    )
    decoded_payload = module.decode_control_payload(payload)
    decoded = decoded_payload.dp_metadata_list

    assert payload.startswith(b"{")
    assert isinstance(decoded[7], AFDDPMetadata)
    assert _tolist(decoded[7].num_tokens_across_dp_cpu) == [3, 5]
    assert int(decoded[7].max_tokens_across_dp_cpu) == 5
    with decoded[7].sp_local_sizes(sequence_parallel_size=1):
        assert decoded[7].get_chunk_sizes_across_dp_rank() == [3, 5]
    assert _tolist(decoded[7].cu_tokens_across_sp(1)) == [3, 8]
    assert decoded_payload.is_graph_capturing is True
    assert decoded_payload.is_warmup is False


def test_p2p_custom_ops_register_send_recv_with_fake_impls(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    calls = []

    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object
    # Empty ops namespace: the registration helper skips ops that already
    # exist on torch.ops.vllm, so the fake must report none registered.
    torch_module.ops = SimpleNamespace(vllm=SimpleNamespace())

    vllm_module = types.ModuleType("vllm")
    utils_module = types.ModuleType("vllm.utils")
    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(**kwargs):
        calls.append(kwargs)

    torch_utils_module.direct_register_custom_op = direct_register_custom_op
    utils_module.torch_utils = torch_utils_module
    vllm_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_module)
    monkeypatch.setattr(module, "torch", torch_module)
    monkeypatch.setattr(module, "direct_register_custom_op", direct_register_custom_op)
    monkeypatch.setattr(module, "_AFD_CUSTOM_OPS_REGISTERED", False)

    module._register_p2p_custom_ops()

    assert [call["op_name"] for call in calls] == [
        "afd_p2p_send",
        "afd_p2p_recv",
    ]
    assert calls[0]["mutates_args"] == ["tensor"]
    assert calls[1]["mutates_args"] == ["out"]
    assert callable(calls[0]["fake_impl"])
    assert callable(calls[1]["fake_impl"])


def test_p2p_hidden_state_send_uses_registered_custom_op(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    connector.a2e_comm_id = 17

    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_send=lambda tensor, dst, comm_id: (
                calls.append((tensor, dst, comm_id)) or None
            ),
        ),
    )
    monkeypatch.setattr(module, "torch", torch_module)

    hidden_states = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(4, 16),
        dtype="bf16",
    )
    output = connector._send_hidden_states(
        hidden_states,
        1,
        SimpleNamespace(world_size=2, rank=0),
        connector.a2e_comm_id,
    )

    assert calls == [(hidden_states, 1, 17)]
    assert output is None


def test_p2p_recv_preserves_dynamic_ref_tensor_first_dim(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    connector.e2a_comm_id = 23

    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_recv=lambda tensor, src, comm_id: (
                calls.append((tensor, src, comm_id)) or None
            ),
        ),
    )
    torch_module.empty = lambda *_args, **_kwargs: pytest.fail(
        "recv should reuse the dynamic ref tensor",
    )
    monkeypatch.setattr(module, "torch", torch_module)

    ref_tensor = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(7, 16),
        dtype="bf16",
    )
    tensor_metadata = SimpleNamespace(
        device="cuda:0",
        dtype="bf16",
        size=(64, 16),
    )

    output = connector._recv_hidden_states(
        0,
        SimpleNamespace(world_size=2, rank=1),
        connector.e2a_comm_id,
        tensor_metadata,
        ref_tensor=ref_tensor,
    )

    assert output is ref_tensor
    assert calls == [(ref_tensor, 0, 23)]


def test_p2p_recv_single_rank_requires_ref_tensor():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    connector.e2a_comm_id = 23
    tensor_metadata = SimpleNamespace(
        device="cuda:0",
        dtype="bf16",
        size=(64, 16),
    )

    with pytest.raises(RuntimeError, match="requires a reference tensor"):
        connector._recv_hidden_states(
            0,
            SimpleNamespace(world_size=1, rank=0),
            connector.e2a_comm_id,
            tensor_metadata,
        )


def test_p2p_async_streams_enforce_compute_and_buffer_dependencies(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(
            use_ubatching=True,
            num_ubatches=2,
            extra_config={"async_transfer": True, "async_slots": 2},
        ),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
    )
    log = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            log.append((self.name, "wait", event.name))

        def synchronize(self):
            log.append((self.name, "synchronize"))

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            log.append((self.name, "record", stream.name))

    class FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            log.append((self.stream.name, "enter"))

        def __exit__(self, *_args):
            log.append((self.stream.name, "exit"))

    class FakeCuda:
        def __init__(self):
            self.compute = FakeStream("compute")
            self.event_count = 0

        def current_stream(self, _device):
            return self.compute

        def Event(self, **_kwargs):  # noqa: N802 - mirrors torch.cuda.Event
            event = FakeEvent(f"event-{self.event_count}")
            self.event_count += 1
            return event

        def stream(self, stream):
            return FakeStreamContext(stream)

    class FakeTensor:
        is_cpu = False
        device = "cuda:0"
        dtype = "bf16"

        def __init__(self, shape=(4, 16)):
            self.shape = tuple(shape)

        def record_stream(self, stream):
            log.append(("tensor", "record_stream", stream.name))

    fake_cuda = FakeCuda()
    torch_module = types.ModuleType("torch")
    torch_module.cuda = fake_cuda
    torch_module.empty = lambda size, **_kwargs: FakeTensor(size)
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_send=lambda _tensor, dst, comm_id: log.append(
                ("send", dst, comm_id),
            ),
            afd_p2p_recv=lambda _tensor, src, comm_id: log.append(
                ("recv", src, comm_id),
            ),
        ),
    )
    monkeypatch.setattr(module, "torch", torch_module)
    connector._a2f_stream = FakeStream("a2f")
    connector._f2a_stream = FakeStream("f2a")
    tensor = FakeTensor()
    process_group = SimpleNamespace(world_size=2, rank=0)

    connector._send_hidden_states(
        tensor,
        1,
        process_group,
        17,
        direction=module._A2F_DIRECTION,
        stage_idx=0,
    )
    output = connector._recv_hidden_states(
        0,
        process_group,
        23,
        SimpleNamespace(device="cuda:0", dtype="bf16", size=(4, 16)),
        ref_tensor=tensor,
        direction=module._F2A_DIRECTION,
        stage_idx=0,
    )

    assert output is tensor
    assert log == [
        ("event-0", "record", "compute"),
        ("a2f", "wait", "event-0"),
        ("a2f", "enter"),
        ("send", 1, 17),
        ("tensor", "record_stream", "a2f"),
        ("event-1", "record", "a2f"),
        ("a2f", "exit"),
        ("f2a", "wait", "event-1"),
        ("f2a", "enter"),
        ("recv", 0, 23),
        ("tensor", "record_stream", "f2a"),
        ("event-2", "record", "f2a"),
        ("f2a", "exit"),
        ("compute", "wait", "event-2"),
        ("tensor", "record_stream", "compute"),
    ]

    log.clear()
    first_buffer = connector._recv_hidden_states(
        1,
        process_group,
        17,
        SimpleNamespace(device="cuda:0", dtype="bf16", size=(4, 16)),
        direction=module._A2F_DIRECTION,
        stage_idx=1,
    )
    connector._record_ffn_input_consumed(first_buffer, stage_idx=1)
    second_buffer = connector._recv_hidden_states(
        1,
        process_group,
        17,
        SimpleNamespace(device="cuda:0", dtype="bf16", size=(8, 16)),
        direction=module._A2F_DIRECTION,
        stage_idx=1,
    )

    assert first_buffer is not second_buffer
    assert second_buffer.shape == (8, 16)
    assert list(connector._async_recv_buffers) == [
        (module._A2F_DIRECTION, 1, 1),
    ]
    input_consumed = connector._async_events[
        (module._INPUT_CONSUMED_EVENT, module._A2F_DIRECTION, 1)
    ]
    assert ("a2f", "wait", input_consumed.name) in log


def test_p2p_async_close_drains_streams_and_clears_resources():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(
            use_ubatching=True,
            num_ubatches=2,
            extra_config={"async_transfer": True},
        ),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
    )
    calls = []
    connector._a2f_stream = SimpleNamespace(
        synchronize=lambda: calls.append("a2f"),
    )
    connector._f2a_stream = SimpleNamespace(
        synchronize=lambda: calls.append("f2a"),
    )
    connector._async_events[("kind", "direction", 0)] = object()
    connector._async_recv_buffers[("a2f", 0, 1)] = object()

    connector.close()

    assert calls == ["a2f", "f2a"]
    assert connector._a2f_stream is None
    assert connector._f2a_stream is None
    assert connector._async_events == {}
    assert connector._async_recv_buffers == {}
    assert connector.is_initialized is False
