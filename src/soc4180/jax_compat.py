"""Make brax and flax work with a current JAX.

JAX 0.11 removed two long-deprecated functions that brax and flax still call:

- ``jax.core.get_opaque_trace_state``  (flax)
- ``jax.device_put_replicated``        (brax)

brax declares only ``jax>=0.4.6``, with no upper bound, so pip cheerfully
installs an incompatible pair — and Colab ships JAX 0.11 preinstalled, so the
combination is broken out of the box.

The alternative fix is to pin ``jax==0.9.2``, the newest release that still has
both. That works, but on Colab it means a large download *and* a runtime restart
on every fresh session, because pip cannot replace a module Python has already
imported. Restoring the two functions is cheaper and leaves Colab's own
GPU-enabled JAX in place.

Both replacements are the ones JAX's own deprecation messages point to.
"""

from __future__ import annotations

__all__ = ["patch_jax"]


def patch_jax(verbose: bool = True) -> list[str]:
    """Restore the removed APIs. Returns the names that were patched.

    Safe to call more than once, and a no-op on a JAX old enough not to need it.
    Call it **before** importing brax or flax.
    """
    import jax
    import jax.numpy as jnp

    patched: list[str] = []

    # flax.core.tracers calls this; JAX moved it to jax.extend.core
    import jax.core as jax_core

    if not hasattr(jax_core, "get_opaque_trace_state"):
        import jax.extend.core as jax_extend_core

        jax_core.get_opaque_trace_state = jax_extend_core.get_opaque_trace_state
        patched.append("jax.core.get_opaque_trace_state")

    # brax's training loop calls this; it was part of the old pmap API
    if not hasattr(jax, "device_put_replicated"):

        def device_put_replicated(x, devices):
            """Stack ``x`` once per device, as the removed function did."""
            stacked = jax.tree.map(
                lambda leaf: jnp.stack([jnp.asarray(leaf)] * len(devices)), x
            )
            if len(devices) == 1:
                return jax.device_put(stacked, devices[0])
            return stacked

        jax.device_put_replicated = device_put_replicated
        patched.append("jax.device_put_replicated")

    if verbose:
        if patched:
            print(f"patched for jax {jax.__version__}: {', '.join(patched)}")
        else:
            print(f"jax {jax.__version__} needs no patching")
    return patched
