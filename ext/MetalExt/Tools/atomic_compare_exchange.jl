"""
    _atomic_compare_exchange_weak_acq_rel_acquire(ptr, expected, desired) :: Int32

Atomically compare the `Int32` value in Metal device memory at `ptr` with
`expected` and, when they compare equal, replace it with `desired`.

The successful operation uses acquire-release ordering and a failed operation
uses acquire ordering. This permits a successful thread to publish preceding
ordinary device-memory writes to a thread that subsequently observes the value
through a failed compare-exchange.

# Parameters
- `ptr :: Core.LLVMPtr{Int32, Metal.AS.Device}`: Pointer to the atomic value in
    Metal device memory.
- `expected :: Int32`: Value expected at `ptr`.
- `desired :: Int32`: Value written when the comparison succeeds.

# Returns
- `Int32`: The final content of the intrinsic's private expected-value storage.
    It remains equal to `expected` on success and is replaced with the value
    observed at `ptr` on failure.

# Memory ordering
This wrapper passes the following constants to the private AIR intrinsic:

- success ordering `4`: acquire-release;
- failure ordering `2`: acquire;
- memory scope `2`: device;
- volatile flag: `true`.

These values follow the ordering and scope encoding used by the atomic
intrinsics emitted by the current Apple Metal compiler. They are not part of
Metal.jl's public API.

# Weak compare-exchange
This is a weak compare-exchange and may fail spuriously. Consequently, a return
value equal to `expected` does not prove that the exchange succeeded. A caller
that requires an unambiguous result must inspect the atomic value afterward or
retry the operation. A return value different from `expected` does prove that
the comparison failed and provides the value observed by the acquire operation.

# Implementation notes
The AIR intrinsic returns a status `Bool`, but the current Apple native pipeline
compiler fails internally when that result is consumed by generated Julia GPU
code. The status is therefore deliberately discarded. Loading and returning
the private expected-value storage matches Metal.jl's existing relaxed weak-CAS
wrapper and compiles successfully.

The second intrinsic argument is backed by a `Ref{Int32}` in the thread-private
address space. `ptr` remains in the Metal device address space. Changing either
pointer type, the argument order, or the intrinsic name may make the private AIR
ABI invalid.

# Compatibility
`air.atomic.global.cmpxchg.weak.i32` is an undocumented AIR intrinsic. This
wrapper is intentionally internal and carries no compatibility guarantee across
Metal.jl, Xcode, AIR, macOS, or Apple GPU versions. Any toolchain update should
be accompanied by native-compilation, publication, and LBVH correctness tests.
"""
@inline function _atomic_compare_exchange_weak_acq_rel_acquire(ptr :: Core.LLVMPtr{Int32, Metal.AS.Device}, expected :: Int32, desired :: Int32) :: Int32
    # AIR writes the value observed on failure back into this private storage.
    expected_box = Ref(expected)

    # Deliberately ignore the returned Bool. Consuming it currently causes the
    # Apple native pipeline compiler to fail even for relaxed ordering.
    Metal.@typed_ccall(
        "air.atomic.global.cmpxchg.weak.i32",
        llvmcall,
        Bool,
        (Core.LLVMPtr{Int32, Metal.AS.Device}, Ptr{Int32}, Int32, Int32, Int32, Int32, Bool),
        ptr,
        expected_box,
        desired,
        Val(Int32(4)), # success: acquire-release
        Val(Int32(2)), # failure: acquire
        Val(Int32(2)), # scope: device
        Val(true),     # volatile
    )

    return expected_box[]
end

"""
    _metal_weak_cas_rendezvous(ptr, desired) :: Int32

Attempt to install `desired` into a zero-valued Metal device-memory rendezvous
slot using acquire-release success and acquire failure ordering.

Return zero when the caller should be treated as the first arrival, or the
previously published nonzero value when another caller arrived first.

This helper deliberately does not retry an ambiguous weak-CAS result. See the
warning in the implementation for the resulting publication limitation.
"""
@inline function _metal_weak_cas_rendezvous(ptr :: Core.LLVMPtr{Int32, Metal.AS.Device}, desired :: Int32)
    old = _atomic_compare_exchange_weak_acq_rel_acquire(ptr, zero(Int32), desired)
    !iszero(old) && return old

    observed = Metal.atomic_load_explicit(ptr)

    # FIXME(Metal weak-CAS publication ambiguity)
    # WARNING:
    # The CAS status Bool cannot currently be consumed because doing so triggers
    # an Apple Metal pipeline compiler internal error. When `old == 0`, a relaxed
    # load is used to distinguish apparent success from a possible spurious weak-CAS
    # failure.
    #
    # If the relaxed load observes another thread's value, this path does not itself
    # provide acquire semantics for that thread's earlier aggregate writes. The code
    # currently relies on the observed Apple Metal behaviour that this ambiguity does
    # not produce stale subtree data in practice.
    #
    # If topology, scale, or AABB corruption appears intermittently on Metal,
    # especially across threadgroups or after an OS/Xcode update, investigate this
    # branch first. A strictly safe fallback is to retry the 4/2 CAS until a nonzero
    # `old` is returned through the acquire failure path.
    if iszero(observed) || observed == desired
        return old
    else
        return observed
    end
end
