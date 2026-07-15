function Partia.Tools._issorted(array :: MtlVector{T}) :: Bool where {T <: Unsigned}
    n = length(array)
    n ≤ 1 && return true

    flag = similar(array, Int32, 1)
    fill!(flag, Int32(0))

    threads = 256
    groups = cld(n - 1, threads)

    @metal threads=threads groups=groups _issorted_kernel!(flag, array, n)

    return only(Array(flag)) == 0
end


@inline function _issorted_kernel!(flag :: MtlDeviceVector{Int32, 1}, array :: MtlDeviceVector{T, 1}, n :: Int) where {T <: Unsigned}
    # Get the global thread index and stride
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    ptr = pointer(flag)
    i = tid
    while i < n
        @inbounds if array[i] > array[i + 1]
            # Multiple threads may write the same aligned Int32 value `1` concurrently.
            # This is intentional: the update is idempotent, no thread writes `0`, and
            # the flag is only read after the kernel has completed.
            unsafe_store!(ptr, Int32(1))
        end

        i += stride
    end
    return nothing
end