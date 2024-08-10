class StoragePtr:
    ptr: str
    uuid: str

def storage_full(fill_value: float, size: int, device: str) -> StoragePtr: ...
def storage_clone(storage_ptr: StoragePtr) -> StoragePtr: ...
def storage_drop(storage_ptr: StoragePtr) -> None: ...
def storage_relu(
    a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int
) -> None: ...
def storage_add(
    a: StoragePtr,
    b: StoragePtr,
    c: StoragePtr,
    idx_a: int,
    idx_b: int,
    idx_c: int,
    size: int,
) -> None: ...
