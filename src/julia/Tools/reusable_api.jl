"""
    build!(storage, args...)

Rebuild an object using already compatible storage. Concrete modules extend
this generic with typed methods and define the required storage invariants.
Methods conventionally return `nothing` after updating `storage` in place.
"""
function build! end

"""
    update!(storage, args...)

Resize reusable storage when necessary and rebuild it from the supplied data.
Concrete modules extend this generic with typed methods. Methods conventionally
return the same reusable wrapper passed as `storage`.
"""
function update! end
