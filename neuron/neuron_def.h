#pragma once
#define CHECK_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x" must be a CUDA tensor"); if (! x.is_contiguous()){x = x.contiguous();}




