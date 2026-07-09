def test_root_cuda_backend_module_imports_without_cupy():
    import backends.cuda_backend as cuda_backend

    assert hasattr(cuda_backend, "GEMM_KERNEL")
    assert hasattr(cuda_backend, "CUDABackend")


def test_legacy_cude_backend_shim_imports():
    import backends.cude_backend as cude_backend

    assert hasattr(cude_backend, "GEMM_KERNEL")
    assert hasattr(cude_backend, "CUDABackend")
