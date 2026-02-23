import jax
import jax.numpy as jnp

from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.strategy.check_early_stop import CheckEarlyStop


def _make_resources(n_chains=10, n_steps=100, global_acc_value=None):
    """Helper to create a minimal set of resources for CheckEarlyStop tests."""
    state = State(
        {
            "target_global_accs": "global_accs_training",
            "training": True,
            "early_stopped": False,
        },
        name="sampler_state",
    )
    buffer = Buffer("global_accs_training", (n_chains, n_steps), 1)
    if global_acc_value is not None:
        # Fill the buffer with the given acceptance rate
        data = jnp.full((n_chains, n_steps), global_acc_value)
        buffer.data = data
    resources = {
        "sampler_state": state,
        "global_accs_training": buffer,
    }
    return resources


class TestCheckEarlyStop:
    """Tests for the CheckEarlyStop strategy."""

    def test_no_trigger_during_warmup(self):
        """Early stop should not trigger during the warmup period."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=3,
            patience=1,
        )
        resources = _make_resources(global_acc_value=0.5)
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Call 3 times (warmup period)
        for _ in range(3):
            key, resources, pos = strategy(key, resources, pos, {})

        state = resources["sampler_state"]
        assert isinstance(state, State)
        assert state.data["early_stopped"] is False

    def test_trigger_when_stable(self):
        """Early stop should trigger when acceptance is stable after warmup."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=2,
            patience=1,
        )
        resources = _make_resources(global_acc_value=0.5)
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Call through warmup (2 loops) + 1 more with same acceptance
        for _ in range(3):
            key, resources, pos = strategy(key, resources, pos, {})

        state = resources["sampler_state"]
        assert isinstance(state, State)
        # Acceptance hasn't changed at all (relative change = 0), should trigger
        assert state.data["early_stopped"] is True

    def test_no_trigger_when_changing(self):
        """Early stop should not trigger when acceptance is changing."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.01,
            n_loops_skip=1,
            patience=1,
        )
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Loop 1: acceptance = 0.3
        resources = _make_resources(global_acc_value=0.3)
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Loop 2: acceptance = 0.6 (big change)
        resources["global_accs_training"].data = jnp.full((10, 100), 0.6)
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

    def test_patience_requires_multiple_stable_loops(self):
        """With patience > 1, early stop requires multiple consecutive stable loops."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=1,
            patience=3,
        )
        resources = _make_resources(global_acc_value=0.5)
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Warmup (1 loop)
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Stable loop 1 — patience 1/3
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Stable loop 2 — patience 2/3
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Stable loop 3 — patience 3/3, should trigger
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is True

    def test_patience_resets_on_unstable(self):
        """Patience counter should reset if acceptance becomes unstable."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=1,
            patience=2,
        )
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Warmup with acceptance = 0.5
        resources = _make_resources(global_acc_value=0.5)
        key, resources, pos = strategy(key, resources, pos, {})

        # Stable loop 1 — patience 1/2
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Unstable loop — big jump, resets patience
        resources["global_accs_training"].data = jnp.full((10, 100), 0.8)
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Stable loop 1 again — patience 1/2
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Stable loop 2 — patience 2/2, should trigger
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is True

    def test_empty_buffer_no_crash(self):
        """Strategy should handle empty (all -inf) buffer gracefully."""
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=1,
            patience=1,
        )
        # Buffer initialized with -inf (default)
        resources = _make_resources(global_acc_value=None)
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Should not crash even with empty buffer
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

    def test_acceptance_window_uses_only_recent_steps(self):
        """acceptance_window should restrict comparison to the last N thinned steps.

        We fill the first 50 columns with 0.1 and the last 50 with 0.9.
        Without windowing, the mean shifts significantly between calls because
        new writes push the overall average.  With acceptance_window=50, each
        call sees only its own 50 columns, so both calls return ~the same
        value and early stopping triggers immediately after warmup.
        """
        n_chains, n_steps = 10, 100
        window = 50  # one loop's worth of global steps

        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            acceptance_window=window,
            n_loops_skip=1,
            patience=1,
        )

        state = State(
            {
                "target_global_accs": "global_accs_training",
                "training": True,
                "early_stopped": False,
            },
            name="sampler_state",
        )
        buffer = Buffer("global_accs_training", (n_chains, n_steps), 1)
        # First half: 0.1, second half: 0.9
        buffer.data = jnp.concatenate(
            [
                jnp.full((n_chains, window), 0.1),
                jnp.full((n_chains, window), 0.9),
            ],
            axis=1,
        )
        resources = {"sampler_state": state, "global_accs_training": buffer}
        key = jax.random.key(0)
        pos = jnp.zeros((n_chains, 2))

        # Warmup call — sees last 50 cols = 0.9, stores prev=0.9
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Second call — still sees last 50 cols = 0.9, relative change = 0 → trigger
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is True

    def test_no_trigger_when_mean_stable_but_cov_changing(self):
        """Early stop should NOT trigger if CoV is changing even when mean is stable.

        Warmup: all chains = 0.5 (uniform → CoV = 0).
        Check:  half chains = 0.3, half = 0.7 (mean still 0.5, CoV >> 0).
        The CoV change is huge, so the stability condition is not met.
        """
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=1,
            patience=1,
        )
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Warmup: uniform acceptance 0.5 across all 10 chains → CoV = 0
        resources = _make_resources(n_chains=10, n_steps=100, global_acc_value=0.5)
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Check: first 5 chains = 0.3, last 5 chains = 0.7 → mean = 0.5, CoV >> 0
        resources["global_accs_training"].data = jnp.concatenate(
            [
                jnp.full((5, 100), 0.3),
                jnp.full((5, 100), 0.7),
            ],
            axis=0,
        )
        key, resources, pos = strategy(key, resources, pos, {})
        # Mean is still 0.5 but CoV jumped → should NOT trigger
        assert resources["sampler_state"].data["early_stopped"] is False

    def test_trigger_when_both_mean_and_cov_stable(self):
        """Early stop should trigger only when both mean AND CoV are stable.

        Use non-uniform acceptance across chains so CoV > 0, but keep both
        mean and CoV exactly the same across loops so both conditions are met.
        """
        strategy = CheckEarlyStop(
            state_name="sampler_state",
            acceptance_buffer_key="target_global_accs",
            relative_tolerance=0.05,
            n_loops_skip=1,
            patience=1,
        )
        key = jax.random.key(0)
        pos = jnp.zeros((10, 2))

        # Non-uniform but fixed: first 5 chains = 0.4, last 5 = 0.6
        # mean = 0.5, CoV = std([0.4*5, 0.6*5]) / 0.5 — constant across loops
        fixed_data = jnp.concatenate(
            [jnp.full((5, 100), 0.4), jnp.full((5, 100), 0.6)], axis=0
        )
        resources = _make_resources(n_chains=10, n_steps=100)
        resources["global_accs_training"].data = fixed_data

        # Warmup: records mean=0.5 and its CoV
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is False

        # Check: same data → both mean change = 0, CoV change = 0 → trigger
        resources["global_accs_training"].data = fixed_data
        key, resources, pos = strategy(key, resources, pos, {})
        assert resources["sampler_state"].data["early_stopped"] is True
