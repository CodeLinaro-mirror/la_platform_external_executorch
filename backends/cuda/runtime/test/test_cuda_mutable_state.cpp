/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#include <executorch/runtime/core/error.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cu = ::executorch::backends::cuda;
namespace aoti = ::executorch::backends::aoti;
namespace slim = ::executorch::backends::aoti::slim;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::runtime::Error;

namespace {

Error fake_get_num_constants(
    aoti::AOTInductorModelContainerHandle,
    size_t* num_constants) {
  *num_constants = 0;
  return Error::Ok;
}

Error fake_get_constant_name(
    aoti::AOTInductorModelContainerHandle,
    size_t,
    const char**) {
  return Error::Ok;
}

Error fake_get_constant_original_fqn(
    aoti::AOTInductorModelContainerHandle,
    size_t,
    const char**) {
  return Error::Ok;
}

Error fake_extract_constants_map(
    aoti::AOTInductorModelContainerHandle,
    aoti::AOTInductorConstantMapHandle,
    bool) {
  return Error::Ok;
}

Error fake_update_user_managed_pairs(
    aoti::AOTInductorModelContainerHandle,
    const aoti::AOTInductorConstantMapEntry*,
    size_t,
    bool,
    bool) {
  return Error::Ok;
}

struct FakeContainer {
  std::vector<std::string> internal_names;
  std::vector<std::string> fqns;
  std::unordered_map<std::string, aoti::AtenTensorHandle> extracted;
  size_t update_calls = 0;
  size_t last_num_pairs = 0;
  std::string last_name;
  void* last_bound_data = nullptr;
  size_t last_bound_nbytes = 0;
};

Error fake_container_get_num_constants(
    aoti::AOTInductorModelContainerHandle container,
    size_t* num_constants) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *num_constants = c->internal_names.size();
  return Error::Ok;
}

Error fake_container_get_constant_name(
    aoti::AOTInductorModelContainerHandle container,
    size_t idx,
    const char** name) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *name =
      idx < c->internal_names.size() ? c->internal_names[idx].c_str() : nullptr;
  return Error::Ok;
}

Error fake_container_get_constant_original_fqn(
    aoti::AOTInductorModelContainerHandle container,
    size_t idx,
    const char** fqn) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  *fqn = idx < c->fqns.size() ? c->fqns[idx].c_str() : nullptr;
  return Error::Ok;
}

Error fake_container_extract_constants_map(
    aoti::AOTInductorModelContainerHandle container,
    aoti::AOTInductorConstantMapHandle map_handle,
    bool) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  auto* out = reinterpret_cast<
      std::unordered_map<std::string, aoti::AtenTensorHandle>*>(map_handle);
  *out = c->extracted;
  return Error::Ok;
}

Error fake_container_update_user_managed_pairs(
    aoti::AOTInductorModelContainerHandle container,
    const aoti::AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool,
    bool) {
  auto* c = reinterpret_cast<FakeContainer*>(container);
  c->update_calls++;
  c->last_num_pairs = num_pairs;
  if (num_pairs > 0) {
    c->last_name = pairs[0].name;
    auto* t = reinterpret_cast<slim::SlimTensor*>(pairs[0].handle);
    c->last_bound_data = t->data_ptr();
    c->last_bound_nbytes = t->nbytes();
  }
  return Error::Ok;
}

cu::CudaDelegateHandle fake_symbol_handle() {
  cu::CudaDelegateHandle handle{};
  handle.get_num_constants = fake_get_num_constants;
  handle.get_constant_name = fake_get_constant_name;
  handle.get_constant_original_fqn = fake_get_constant_original_fqn;
  handle.extract_constants_map = fake_extract_constants_map;
  handle.update_user_managed_constant_buffer_pairs =
      fake_update_user_managed_pairs;
  return handle;
}

cu::CudaDelegateHandle fake_container_handle(FakeContainer* container) {
  cu::CudaDelegateHandle handle{};
  handle.container_handle =
      reinterpret_cast<aoti::AOTInductorModelContainerHandle>(container);
  handle.get_num_constants = fake_container_get_num_constants;
  handle.get_constant_name = fake_container_get_constant_name;
  handle.get_constant_original_fqn = fake_container_get_constant_original_fqn;
  handle.extract_constants_map = fake_container_extract_constants_map;
  handle.update_user_managed_constant_buffer_pairs =
      fake_container_update_user_managed_pairs;
  return handle;
}

bool cuda_device_available() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

std::unique_ptr<slim::SlimTensor> make_device_tensor(
    const std::vector<float>& values,
    void** device_ptr) {
  *device_ptr = nullptr;
  cudaError_t err = cudaMalloc(device_ptr, values.size() * sizeof(float));
  if (err != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc failed: " << cudaGetErrorString(err);
    return nullptr;
  }
  err = cudaMemcpy(
      *device_ptr,
      values.data(),
      values.size() * sizeof(float),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    ADD_FAILURE() << "cudaMemcpy failed: " << cudaGetErrorString(err);
    cudaFree(*device_ptr);
    *device_ptr = nullptr;
    return nullptr;
  }
  return std::make_unique<slim::SlimTensor>(slim::from_blob(
      *device_ptr,
      {static_cast<int64_t>(values.size())},
      slimc10::ScalarType::Float,
      slimc10::Device(slimc10::DeviceType::CUDA, 0)));
}

std::unique_ptr<slim::SlimTensor> make_cpu_tensor(std::vector<float>& values) {
  return std::make_unique<slim::SlimTensor>(slim::from_blob(
      values.data(),
      {static_cast<int64_t>(values.size())},
      slimc10::ScalarType::Float,
      slimc10::Device(slimc10::DeviceType::CPU, 0)));
}

} // namespace

TEST(CudaMutableStateTest, FallClosedDefaults) {
  const cu::MutableStateContext bad = 999999;
  cu::MutableStateContext c1 = cu::mutable_state_create_context();
  cu::MutableStateContext c2 = cu::mutable_state_create_context();

  EXPECT_GT(c2, c1);
  EXPECT_FALSE(cu::mutable_state_available(c1));
  EXPECT_EQ(cu::mutable_state_bytes_per_session(c1), 0);
  EXPECT_EQ(cu::mutable_state_bytes_per_session(bad), 0);
  EXPECT_EQ(cu::mutable_state_validate_coverage(bad), Error::InvalidArgument);
  EXPECT_EQ(cu::mutable_state_validate_coverage(c1), Error::NotSupported);

  cu::mutable_state_register_fqns(c1, {"a.b", "c.d"});
  EXPECT_EQ(cu::mutable_state_validate_coverage(c1), Error::NotSupported);
  EXPECT_EQ(
      cu::mutable_state_create_session(bad).error(), Error::InvalidArgument);
  EXPECT_EQ(cu::mutable_state_create_session(c1).error(), Error::NotSupported);

  cu::mutable_state_destroy_session(bad, 0);
  cu::mutable_state_destroy_context(bad);
  cu::mutable_state_destroy_context(c1);
  cu::mutable_state_destroy_context(c2);
}

TEST(CudaMutableStateTest, ForgetHandleDropsAssociation) {
  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::CudaDelegateHandle handle{};

  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&handle);
  cu::mutable_state_end_load();

  cu::mutable_state_set_active(c, 0);
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::NotSupported);

  cu::mutable_state_forget_handle(&handle);
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::Internal);

  cu::mutable_state_set_active(
      cu::kInvalidMutableContext, cu::kNoMutableSession);
  cu::mutable_state_destroy_context(c);
}

TEST(CudaMutableStateTest, RebindRejectsUncreatedSessionToken) {
  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&handle);
  cu::mutable_state_end_load();
  ASSERT_TRUE(cu::mutable_state_available(c));
  ASSERT_EQ(cu::mutable_state_validate_coverage(c), Error::Ok);

  cu::mutable_state_set_active(c, 123);
  EXPECT_EQ(
      cu::mutable_state_rebind_for_execute(&handle), Error::InvalidArgument);

  auto token = cu::mutable_state_create_session(c);
  ASSERT_TRUE(token.ok());
  cu::mutable_state_set_active(c, token.get());
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::Internal);

  cu::mutable_state_set_active(
      cu::kInvalidMutableContext, cu::kNoMutableSession);
  cu::mutable_state_destroy_session(c, token.get());
  cu::mutable_state_destroy_context(c);
}

TEST(CudaMutableStateTest, NestedBeginLoadFailsClosed) {
  cu::MutableStateContext c1 = cu::mutable_state_create_context();
  cu::MutableStateContext c2 = cu::mutable_state_create_context();
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  cu::mutable_state_begin_load(c1);
  cu::mutable_state_begin_load(c2);
  cu::mutable_state_note_handle(&handle);
  cu::mutable_state_end_load();

  EXPECT_EQ(cu::mutable_state_validate_coverage(c1), Error::InvalidState);
  EXPECT_EQ(cu::mutable_state_validate_coverage(c2), Error::InvalidState);
  EXPECT_FALSE(cu::mutable_state_available(c1));
  EXPECT_FALSE(cu::mutable_state_available(c2));
  EXPECT_EQ(cu::mutable_state_create_session(c1).error(), Error::InvalidState);
  EXPECT_EQ(cu::mutable_state_create_session(c2).error(), Error::InvalidState);

  cu::mutable_state_destroy_context(c1);
  cu::mutable_state_destroy_context(c2);
}

TEST(CudaMutableStateTest, RebindRejectsCudaGraphHandle) {
  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::CudaDelegateHandle handle = fake_symbol_handle();

  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&handle);
  cu::mutable_state_end_load();
  ASSERT_TRUE(cu::mutable_state_available(c));
  ASSERT_EQ(cu::mutable_state_validate_coverage(c), Error::Ok);

  auto token = cu::mutable_state_create_session(c);
  ASSERT_TRUE(token.ok());

  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  cu::mutable_state_set_active(c, token.get());
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::NotSupported);

  cu::mutable_state_set_active(
      cu::kInvalidMutableContext, cu::kNoMutableSession);
  cu::mutable_state_destroy_session(c, token.get());
  cu::mutable_state_destroy_context(c);
}

TEST(CudaMutableStateTest, CapturesClonesAndRebindsDeviceBuffer) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* source_ptr = nullptr;
  auto source_tensor =
      make_device_tensor({1.0f, 2.0f, 3.0f, 4.0f}, &source_ptr);
  ASSERT_NE(source_tensor, nullptr);
  ASSERT_NE(source_ptr, nullptr);

  FakeContainer container;
  container.internal_names = {"internal_state"};
  container.fqns = {"model.state"};
  container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(source_tensor.get());
  cu::CudaDelegateHandle handle = fake_container_handle(&container);

  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::mutable_state_register_fqns(c, {"model.state"});
  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&handle);
  cu::mutable_state_end_load();

  ASSERT_TRUE(cu::mutable_state_available(c));
  EXPECT_EQ(cu::mutable_state_bytes_per_session(c), 4 * sizeof(float));
  ASSERT_EQ(cu::mutable_state_validate_coverage(c), Error::Ok);

  auto token = cu::mutable_state_create_session(c);
  ASSERT_TRUE(token.ok());
  cu::mutable_state_set_active(c, token.get());
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&handle), Error::Ok);

  EXPECT_EQ(container.update_calls, 1u);
  EXPECT_EQ(container.last_num_pairs, 1u);
  EXPECT_EQ(container.last_name, "internal_state");
  ASSERT_NE(container.last_bound_data, nullptr);
  EXPECT_NE(container.last_bound_data, source_ptr);
  EXPECT_EQ(container.last_bound_nbytes, 4 * sizeof(float));

  std::vector<float> cloned(4);
  EXPECT_EQ(
      cudaMemcpy(
          cloned.data(),
          container.last_bound_data,
          cloned.size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cloned, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

  cu::mutable_state_set_active(
      cu::kInvalidMutableContext, cu::kNoMutableSession);
  cu::mutable_state_destroy_session(c, token.get());
  cu::mutable_state_destroy_context(c);
  cudaFree(source_ptr);
}

TEST(CudaMutableStateTest, SharedFqnAcrossHandlesUsesSameSessionBuffer) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* prefill_ptr = nullptr;
  void* decode_ptr = nullptr;
  auto prefill_tensor = make_device_tensor({1.0f, 2.0f}, &prefill_ptr);
  auto decode_tensor = make_device_tensor({9.0f, 8.0f}, &decode_ptr);
  ASSERT_NE(prefill_tensor, nullptr);
  ASSERT_NE(decode_tensor, nullptr);
  ASSERT_NE(prefill_ptr, nullptr);
  ASSERT_NE(decode_ptr, nullptr);

  FakeContainer prefill_container;
  prefill_container.internal_names = {"prefill_internal_kv"};
  prefill_container.fqns = {"model.kv"};
  prefill_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(prefill_tensor.get());
  cu::CudaDelegateHandle prefill_handle =
      fake_container_handle(&prefill_container);

  FakeContainer decode_container;
  decode_container.internal_names = {"decode_internal_kv"};
  decode_container.fqns = {"model.kv"};
  decode_container.extracted["model.kv"] =
      reinterpret_cast<aoti::AtenTensorHandle>(decode_tensor.get());
  cu::CudaDelegateHandle decode_handle =
      fake_container_handle(&decode_container);

  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::mutable_state_register_fqns(c, {"model.kv"});
  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&prefill_handle);
  cu::mutable_state_note_handle(&decode_handle);
  cu::mutable_state_end_load();

  ASSERT_TRUE(cu::mutable_state_available(c));
  ASSERT_EQ(cu::mutable_state_validate_coverage(c), Error::Ok);

  auto token = cu::mutable_state_create_session(c);
  ASSERT_TRUE(token.ok());
  cu::mutable_state_set_active(c, token.get());
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&prefill_handle), Error::Ok);
  EXPECT_EQ(cu::mutable_state_rebind_for_execute(&decode_handle), Error::Ok);

  ASSERT_NE(prefill_container.last_bound_data, nullptr);
  ASSERT_NE(decode_container.last_bound_data, nullptr);
  EXPECT_EQ(prefill_container.last_name, "prefill_internal_kv");
  EXPECT_EQ(decode_container.last_name, "decode_internal_kv");
  EXPECT_EQ(
      prefill_container.last_bound_data, decode_container.last_bound_data);
  EXPECT_NE(prefill_container.last_bound_data, prefill_ptr);
  EXPECT_NE(decode_container.last_bound_data, decode_ptr);

  cu::mutable_state_set_active(
      cu::kInvalidMutableContext, cu::kNoMutableSession);
  cu::mutable_state_destroy_session(c, token.get());
  cu::mutable_state_destroy_context(c);
  cudaFree(prefill_ptr);
  cudaFree(decode_ptr);
}

TEST(
    CudaMutableStateTest,
    ValidateCoverageRejectsLargerDescriptorForSharedFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* small_ptr = nullptr;
  void* large_ptr = nullptr;
  auto small_tensor = make_device_tensor({1.0f}, &small_ptr);
  auto large_tensor = make_device_tensor({1.0f, 2.0f}, &large_ptr);
  ASSERT_NE(small_tensor, nullptr);
  ASSERT_NE(large_tensor, nullptr);
  ASSERT_NE(small_ptr, nullptr);
  ASSERT_NE(large_ptr, nullptr);

  FakeContainer small_container;
  small_container.internal_names = {"small_internal"};
  small_container.fqns = {"model.state"};
  small_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(small_tensor.get());
  cu::CudaDelegateHandle small_handle = fake_container_handle(&small_container);

  FakeContainer large_container;
  large_container.internal_names = {"large_internal"};
  large_container.fqns = {"model.state"};
  large_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(large_tensor.get());
  cu::CudaDelegateHandle large_handle = fake_container_handle(&large_container);

  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::mutable_state_register_fqns(c, {"model.state"});
  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&small_handle);
  cu::mutable_state_note_handle(&large_handle);
  cu::mutable_state_end_load();

  ASSERT_TRUE(cu::mutable_state_available(c));
  EXPECT_EQ(cu::mutable_state_validate_coverage(c), Error::InvalidProgram);
  EXPECT_FALSE(cu::mutable_state_available(c));
  EXPECT_EQ(cu::mutable_state_create_session(c).error(), Error::InvalidProgram);
  EXPECT_EQ(large_container.update_calls, 0u);

  cu::mutable_state_destroy_context(c);
  cudaFree(small_ptr);
  cudaFree(large_ptr);
}

TEST(CudaMutableStateTest, ValidateCoverageRejectsDeviceMismatchForSharedFqn) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  void* cuda_ptr = nullptr;
  auto cuda_tensor = make_device_tensor({1.0f}, &cuda_ptr);
  ASSERT_NE(cuda_tensor, nullptr);
  ASSERT_NE(cuda_ptr, nullptr);

  std::vector<float> cpu_values = {1.0f};
  auto cpu_tensor = make_cpu_tensor(cpu_values);
  ASSERT_NE(cpu_tensor, nullptr);

  FakeContainer cuda_container;
  cuda_container.internal_names = {"cuda_internal"};
  cuda_container.fqns = {"model.state"};
  cuda_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(cuda_tensor.get());
  cu::CudaDelegateHandle cuda_handle = fake_container_handle(&cuda_container);

  FakeContainer cpu_container;
  cpu_container.internal_names = {"cpu_internal"};
  cpu_container.fqns = {"model.state"};
  cpu_container.extracted["model.state"] =
      reinterpret_cast<aoti::AtenTensorHandle>(cpu_tensor.get());
  cu::CudaDelegateHandle cpu_handle = fake_container_handle(&cpu_container);

  cu::MutableStateContext c = cu::mutable_state_create_context();
  cu::mutable_state_register_fqns(c, {"model.state"});
  cu::mutable_state_begin_load(c);
  cu::mutable_state_note_handle(&cuda_handle);
  cu::mutable_state_note_handle(&cpu_handle);
  cu::mutable_state_end_load();

  ASSERT_TRUE(cu::mutable_state_available(c));
  EXPECT_EQ(cu::mutable_state_validate_coverage(c), Error::InvalidProgram);
  EXPECT_FALSE(cu::mutable_state_available(c));
  EXPECT_EQ(cu::mutable_state_create_session(c).error(), Error::InvalidProgram);
  EXPECT_EQ(cpu_container.update_calls, 0u);

  cu::mutable_state_destroy_context(c);
  cudaFree(cuda_ptr);
}
