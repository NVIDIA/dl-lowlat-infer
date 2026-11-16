/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <nvlstm_inf/CudaEnv.cuh>
#include <nvlstm_inf/ModelAsSingleStepGemv.cuh>
#include <nvlstm_inf/ModelSingleStep.cuh>
#include <nvlstm_inf/ModelSingleStepGemv.cuh>
#include <nvlstm_inf/cuda_helpers.cuh>
#include <utility>

#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

// need to reimplement after the changes
#define TEST_DISABLE_SINGLE_STEP_GEMV_TESTING

using namespace nvidia::quant::lowlat_infer;

std::vector<float> get_random_vector(size_t vector_size, float max_val = 0.5F) {
  std::vector<float> res(vector_size);
  for (size_t i = 0; i < vector_size; ++i) {
    res[i] = (static_cast<float>(rand()) - static_cast<float>(RAND_MAX) * 0.5F) / RAND_MAX * (max_val / 0.5F);
  }
  return res;
}

class ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    cuda_env_ = std::make_unique<CudaEnv>(lenient, create_stream);
    device_count_ = cuda_env_->DeviceCount();
  }

  void TearDown() override { cuda_env_.reset(); }

  std::unique_ptr<CudaEnv> cuda_env_{};
  int device_count_{};
};

#ifndef TEST_DISABLE_SINGLE_STEP_GEMV_TESTING
class ModelAsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CuCheck(cuDeviceGet(&current_device_, 0));
    CuCheck(cuDevicePrimaryCtxRetain(&primary_ctx_, current_device_));
    auto constexpr lenient = true;
    auto constexpr create_stream = true;
    cuda_env_ = std::make_unique<CudaEnv>(lenient, create_stream);
    device_count_ = cuda_env_->DeviceCount();
  }

  void TearDown() override {
    cuda_env_.reset();
    contexts_.resize(0);
    CuCheck(cuDevicePrimaryCtxRelease(current_device_));
  }

  std::unique_ptr<CudaEnv> cuda_env_{};
  int device_count_{};
  CUdevice current_device_;
  CUcontext primary_ctx_;
  GreenContextsVector contexts_;
};
#endif

class ModelParamTest : public ModelTest, public ::testing::WithParamInterface<int> {};

TEST_F(ModelTest, Construct) {
  if (device_count_ == 0) GTEST_SKIP();

  auto path = std::filesystem::path(TEST_RESOURCES_DIR) / "lstm_s";
  ASSERT_TRUE(std::filesystem::exists(path));
  ModelDef model_def{path.c_str()};

  ModelSingleStep model{*cuda_env_, model_def, 1};
  EXPECT_EQ(1, model.BatchSize());
  EXPECT_EQ(model_def.NumSteps(), model.NumSteps());
  EXPECT_EQ(model_def.NumFeatures(), model.NumFeatures());
}

namespace {

template <typename TModel>
void TestSingleNeuronOneStep(CudaEnv &cuda_env, bool single_step) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto constexpr num_steps = 1;
  auto constexpr num_features = 1;
  auto constexpr batch_size = 1;
  util::Tensor32 W{util::Vector32(4, 1.0f), util::ShapeVector{1, 4, 1}};
  util::Tensor32 R{util::Vector32(4, 0.0f), util::ShapeVector{1, 4, 1}};
  util::Tensor32 B{util::Vector32(8, 0.0f), util::ShapeVector{1, 8}};
  util::Tensor32 mat_mul{util::Vector32(1, 1.0f), util::ShapeVector{1, 1}};
  auto constexpr bias = 0.0f;

  ModelDef model_def{
      num_steps,
      num_features,
      util::MakeVector(std::move(W)),
      util::MakeVector(std::move(R)),
      util::MakeVector(std::move(B)),
      std::move(mat_mul),
      bias};

  ModelConfig config{};
  TModel model{cuda_env, model_def, batch_size, config};
  util::Tensor32 const empty{util::Vector32{}, util::ShapeVector{0, num_features}};
  model.InitX(empty);
  auto input = cex::MallocHost<float>(num_features * batch_size);
  input[0.0] = 1.0;
  auto output = cex::MallocHost<float>(batch_size);
  model.Infer(input.get());
  model.Output(output.get());
  EXPECT_NEAR(output[0.0], 0.36960635, 5e-4);

  auto &mW = model_def.LstmW().front();
  std::array const gates{'f', 'i', 'o', 'c'};
  std::array const expected{
      0.369606352935706f,
      0.265666310532337f,
      0.181699742194526f,
      0.0f,
  };
  for (std::size_t i = 0; i < gates.size(); ++i) {
    mW.Data()[model_def.OffsetW(gates[i], true)] = 0;
    TModel mod{cuda_env, model_def, batch_size, config};
    mod.InitX(empty);
    mod.Infer(input.get());
    mod.Output(output.get());
    EXPECT_NEAR(output[0], expected[i], 5e-4);
  }
}
}  // namespace

TEST_F(ModelTest, SingleNeuronOneStepCustom) { TestSingleNeuronOneStep<ModelSingleStep<float>>(*cuda_env_, true); }

namespace {

template <typename TModel>
void TestSingleNeuronTwoSteps(CudaEnv &cuda_env, bool single_step) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto constexpr num_steps = 2;
  auto constexpr num_features = 1;
  auto constexpr batch_size = 1;
  util::Tensor32 W{util::Vector32(4, 1.0f), util::ShapeVector{1, 4, 1}};
  util::Tensor32 R{util::Vector32(4, 1.0f), util::ShapeVector{1, 4, 1}};
  util::Tensor32 B{util::Vector32(8, 0.0f), util::ShapeVector{1, 8}};
  util::Tensor32 mat_mul{util::Vector32(1, 1.0f), util::ShapeVector{1, 1}};
  auto constexpr bias = 0.0f;

  ModelDef model_def{
      num_steps,
      num_features,
      util::MakeVector(std::move(W)),
      util::MakeVector(std::move(R)),
      util::MakeVector(std::move(B)),
      std::move(mat_mul),
      bias};

  ModelConfig config{};
  TModel model{cuda_env, model_def, batch_size, config};
  util::Tensor32 const initial{util::Vector32{1.0f}, util::ShapeVector{1, num_features}};
  model.InitX(initial);
  auto input = cex::MallocHost<float>(num_features * batch_size);
  input[0] = 1.0;
  auto output = cex::MallocHost<float>(batch_size);
  model.Infer(input.get());
  model.Output(output.get());
  EXPECT_NEAR(output[0], 0.650535223200813f, 5e-4);

  auto &mW = model_def.LstmW().front();
  std::array const gates{'f', 'i', 'o', 'c'};
  std::array const expected{
      0.616978887294079f,
      0.470489276557255f,
      0.315073285722705f,
      0.0f,
  };
  for (std::size_t i = 0; i < gates.size(); ++i) {
    mW.Data()[model_def.OffsetW(gates[i], true)] = 0;
    TModel mod{cuda_env, model_def, batch_size, config};
    mod.InitX(initial);
    mod.Infer(input.get());
    mod.Output(output.get());
    EXPECT_NEAR(output[0], expected[i], 5e-4);
  }
}

}  // namespace

TEST_F(ModelTest, SingleNeuronTwoStepsCustom) { TestSingleNeuronTwoSteps<ModelSingleStep<float>>(*cuda_env_, true); }

namespace {

template <typename TModel>
void TestTwoNeurons(CudaEnv &cuda_env, bool single_step) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto constexpr num_steps = 1;
  auto constexpr num_features = 2;
  auto constexpr batch_size = 1;
  util::Tensor32 W{util::Vector32(8, 1.0f), util::ShapeVector{1, 4, 2}};
  util::Tensor32 R{util::Vector32(4, 0.0f), util::ShapeVector{1, 4, 1}};
  util::Tensor32 B{util::Vector32(8, 0.0f), util::ShapeVector{1, 8}};
  util::Tensor32 mat_mul{util::Vector32(1, 1.0f), util::ShapeVector{1, 1}};
  auto constexpr bias = 0.0f;

  ModelDef model_def{
      num_steps,
      num_features,
      util::MakeVector(std::move(W)),
      util::MakeVector(std::move(R)),
      util::MakeVector(std::move(B)),
      std::move(mat_mul),
      bias};

  ModelConfig config{};
  TModel model{cuda_env, model_def, batch_size, config};
  util::Tensor32 const empty{util::Vector32{}, util::ShapeVector{0, num_features}};
  model.InitX(empty);
  auto input = cex::MallocHost<float>(num_features * batch_size);
  input[0] = 1.0;
  input[1] = 0.5;
  auto output = cex::MallocHost<float>(batch_size);
  model.Infer(input.get());
  model.Output(output.get());
  EXPECT_NEAR(output[0], 0.51438592253464f, 5e-4);

  auto &mW = model_def.LstmW().front();
  std::array const gates{'f', 'i', 'o', 'c'};
  std::array const expected{
      0.51438592253464f,
      0.346661912489571f,
      0.212006320270365f,
      0.0f,
  };
  for (std::size_t i = 0; i < gates.size(); ++i) {
    size_t offset = model_def.OffsetW(gates[i], true);
    mW.Data()[offset] = 0;
    mW.Data()[offset + 1] = 0;
    TModel mod{cuda_env, model_def, batch_size, config};
    mod.InitX(empty);
    mod.Infer(input.get());
    mod.Output(output.get());
    EXPECT_NEAR(output[0], expected[i], 5e-4);
  }
}

}  // namespace

TEST_F(ModelTest, TwoNeuronsCustom) { TestTwoNeurons<ModelSingleStep<float>>(*cuda_env_, true); }

namespace {

#ifndef TEST_DISABLE_SINGLE_STEP_GEMV_TESTING
template <typename ModelT>
void TestLstmS(CudaEnv &cuda_env, std::uint32_t batch_size, float abs_error, ModelConfig config = {}) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / "lstm_s").c_str()};

  auto model_ptr = new ModelT{cuda_env, model_def, batch_size, config};
  ModelT &model = *model_ptr;
  auto const data = util::Tensor32::Load(test_resources_path / "lstm_s.npy").Reallocate<cex::PinnedAllocator<float>>();

  auto const num_features = model_def.NumFeatures();
  ASSERT_EQ(data.Shape().at(1), num_features);

  // See test/python/onnx_inf_test.py
  std::array const expected{0.0607472, 0.0581173, 0.0436738, 0.0270372, 0.0463703, 0.0400087, 0.0746033, 0.0671775,
                            0.0712779, 0.0790683, 0.0576262, 0.0523330, 0.0442828, 0.0324383, 0.0257568, 0.0256172,
                            0.0494044, 0.0554544, 0.0564890, 0.0806386, 0.0839941, 0.0799104, 0.0908234, 0.0892975,
                            0.0764109, 0.0825508, 0.0771644, 0.0662445, 0.0850844, 0.0828390, 0.0482092, 0.0491440,
                            0.0636141, 0.0674123, 0.0545414, 0.0378045, 0.0353143, 0.0549297, 0.0599470, 0.0471048,
                            0.0444547, 0.0516035, 0.0402506, 0.0495917, 0.0505177, 0.0569366, 0.0602348, 0.0572838,
                            0.0637554, 0.0633601, 0.0691239, 0.0745556, 0.0537966, 0.0564788, 0.0480169, 0.0828468,
                            0.0670837, 0.0555913, 0.0395631, 0.0364232, 0.0576609, 0.0709375, 0.0831929, 0.0890888,
                            0.0827010, 0.0824412, 0.0921332, 0.0806334, 0.0963287, 0.1083968, 0.0920052, 0.0634612};

  auto const num_steps = model_def.NumSteps();
  auto const init_size = (num_steps - 1) * num_features;

  auto const expected_size = expected.size();
  ASSERT_LE(expected_size, data.Shape().at(0) - num_steps + 1);
  ASSERT_EQ(expected_size % batch_size, 0);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model.InitX(initial);

  auto input = cex::MallocHost<float const *>(batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  auto const num_batches = expected_size / batch_size;
  for (std::size_t batch = 0; batch < num_batches; ++batch) {
    auto const batch_start = batch * batch_size;
    for (std::size_t i = 0; i < batch_size; ++i) {
      input[i] = data_start + (batch_start + i) * num_features;
    }
    model.PreProcess();
    model.Infer(input.get());
    model.Output(output.get(), false, true);
    model.OutputWait();
    model.PostProcess();
    fflush(stdout);
    for (std::size_t offset = 0; offset < batch_size; ++offset) {
      EXPECT_NEAR(output[offset], expected[batch_start + offset], abs_error)
          << " batch: " << batch << "/" << num_batches << " offset: " << offset << "/" << batch_size;
    }
  }

  // AS kernels need this
  delete model_ptr;
  // debug
  // model.~ModelT();
}

template <typename ModelT>
void TestLstmM(CudaEnv &cuda_env, std::uint32_t batch_size, float abs_error, ModelConfig config = {}) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / "lstm_m").c_str()};

  auto model_ptr = new ModelT{cuda_env, model_def, batch_size, config};
  ModelT &model = *model_ptr;
  auto const data = util::Tensor32::Load(test_resources_path / "lstm_m.npy").Reallocate<cex::PinnedAllocator<float>>();

  auto const num_features = model_def.NumFeatures();
  ASSERT_EQ(data.Shape().at(1), num_features);

  // See test/python/onnx_inf_test.py
  std::array const expected{
      -0.115482, -0.11391,  -0.109962, -0.106429, -0.102976, -0.103491, -0.103407, -0.106292, -0.109379, -0.110781,
      -0.112505, -0.109176, -0.106475, -0.106454, -0.106188, -0.104424, -0.105075, -0.106564, -0.107202, -0.106034,
      -0.102799, -0.102786, -0.104316, -0.103755, -0.103805, -0.104216, -0.10373,  -0.101618, -0.099636, -0.098746,
      -0.097549, -0.098384, -0.09808,  -0.100138, -0.102327, -0.100497, -0.099913, -0.101717, -0.104313, -0.105751,
      -0.106392, -0.104682, -0.103746, -0.101673, -0.09999,  -0.099901, -0.102508, -0.103734, -0.10431,  -0.105292,
      -0.105754, -0.104938, -0.107077, -0.109832, -0.113169, -0.116392, -0.117326, -0.115867, -0.112157, -0.108776,
      -0.106809, -0.106665, -0.107101, -0.10728,  -0.10968,  -0.110215, -0.108624, -0.107113, -0.106606, -0.104836,
      -0.103446, -0.099937, -0.098563, -0.096026, -0.096032, -0.097297, -0.098636, -0.100789, -0.100811, -0.10134,
      -0.102574, -0.104604, -0.106295, -0.109517, -0.109754, -0.106337, -0.105459, -0.103619, -0.103879, -0.104555,
      -0.104304, -0.106182, -0.1077,   -0.107955, -0.104887, -0.1028,   -0.102936, -0.104754, -0.108918, -0.109058,
      -0.109621, -0.111552, -0.11092,  -0.10979,  -0.106241, -0.099871, -0.095659, -0.093566, -0.097233, -0.103468,
      -0.107334, -0.108297, -0.109324, -0.109405, -0.106693, -0.104452, -0.104951, -0.104034, -0.103549, -0.103365,
      -0.101367, -0.10493,  -0.106295, -0.1041,   -0.102505, -0.101257, -0.102145, -0.10304};

  auto const num_steps = model_def.NumSteps();
  auto const init_size = (num_steps - 1) * num_features;

  auto const expected_size = expected.size();
  ASSERT_LE(expected_size, data.Shape().at(0) - num_steps + 1);
  ASSERT_EQ(expected_size % batch_size, 0);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model.InitX(initial);

  auto input = cex::MallocHost<float const *>(batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  auto const num_batches = expected_size / batch_size;
  for (std::size_t batch = 0; batch < num_batches; ++batch) {
    auto const batch_start = batch * batch_size;
    for (std::size_t i = 0; i < batch_size; ++i) {
      input[i] = data_start + (batch_start + i) * num_features;
    }
    model.PreProcess();
    model.Infer(input.get());
    model.Output(output.get(), false, true);
    model.OutputWait();
    model.PostProcess();
    fflush(stdout);
    for (std::size_t offset = 0; offset < batch_size; ++offset) {
      EXPECT_NEAR(output[offset], expected[batch_start + offset], abs_error)
          << " batch: " << batch << "/" << num_batches << " offset: " << offset << "/" << batch_size;
    }
  }

  // AS kernels need this
  delete model_ptr;
}
#endif

inline constexpr std::array LSTM_S_EXPECTED{
    0.0607472, 0.0581173, 0.0436738, 0.0270372, 0.0463703, 0.0400087, 0.0746033, 0.0671775, 0.0712779,
    0.0790683, 0.0576262, 0.0523330, 0.0442828, 0.0324383, 0.0257568, 0.0256172, 0.0494044, 0.0554544,
    0.0564890, 0.0806386, 0.0839941, 0.0799104, 0.0908234, 0.0892975, 0.0764109, 0.0825508, 0.0771644,
    0.0662445, 0.0850844, 0.0828390, 0.0482092, 0.0491440, 0.0636141, 0.0674123, 0.0545414, 0.0378045,
    0.0353143, 0.0549297, 0.0599470, 0.0471048, 0.0444547, 0.0516035, 0.0402506, 0.0495917, 0.0505177,
    0.0569366, 0.0602348, 0.0572838, 0.0637554, 0.0633601, 0.0691239, 0.0745556, 0.0537966, 0.0564788,
    0.0480169, 0.0828468, 0.0670837, 0.0555913, 0.0395631, 0.0364232, 0.0576609, 0.0709375, 0.0831929,
    0.0890888, 0.0827010, 0.0824412, 0.0921332, 0.0806334, 0.0963287, 0.1083968, 0.0920052, 0.0634612};

inline constexpr std::array LSTM_S_EXPECTED_STREAMING{
    0.05223169922828674,  0.03289639577269554,  0.028326954692602158, 0.04369994252920151,  0.032637517899274826,
    0.031058330088853836, 0.01778576150536537,  0.03330153971910477,  0.04557318985462189,  0.03518157824873924,
    0.027971778064966202, 0.03884872794151306,  0.05041683837771416,  0.04328927397727966,  0.05617979168891907,
    0.07906169444322586,  0.0770992562174797,   0.06741693615913391,  0.06397736072540283,  0.0536026768386364,
    0.04216962307691574,  0.03283233940601349,  0.035600464791059494, 0.03930624574422836,  0.045927744358778,
    0.04047013074159622,  0.03670214116573334,  0.02564188838005066,  0.02963549643754959,  0.060979850590229034,
    0.06543806940317154,  0.04179079830646515,  0.017726175487041473, 0.02090279757976532,  0.02978019043803215,
    0.03233351185917854,  0.03917557746171951,  0.026869148015975952, 0.05261100456118584,  0.06739369034767151,
    0.06430751830339432,  0.07248286157846451,  0.07301109284162521,  0.10827194154262543,  0.10741305351257324,
    0.10188965499401093,  0.09199462831020355,  0.07713824510574341,  0.07778257131576538,  0.06170262023806572,
    0.06208455190062523,  0.059834614396095276, 0.0664798766374588,   0.06382978707551956,  0.075422503054142,
    0.06407451629638672,  0.050365038216114044, 0.0533042848110199,   0.058045461773872375, 0.06609512120485306,
    0.07025840878486633,  0.07232921570539474,  0.06068526208400726,  0.06074712797999382,  0.058117419481277466,
    0.043674081563949585, 0.02703709527850151,  0.046370282769203186, 0.04000874608755112,  0.07460340857505798,
    0.06717769801616669,  0.07127795368432999};

// See test/python/onnx_inf_test.py
inline constexpr std::array const LSTM_M_EXPECTED{
    -0.115482, -0.11391,  -0.109962, -0.106429, -0.102976, -0.103491, -0.103407, -0.106292, -0.109379, -0.110781,
    -0.112505, -0.109176, -0.106475, -0.106454, -0.106188, -0.104424, -0.105075, -0.106564, -0.107202, -0.106034,
    -0.102799, -0.102786, -0.104316, -0.103755, -0.103805, -0.104216, -0.10373,  -0.101618, -0.099636, -0.098746,
    -0.097549, -0.098384, -0.09808,  -0.100138, -0.102327, -0.100497, -0.099913, -0.101717, -0.104313, -0.105751,
    -0.106392, -0.104682, -0.103746, -0.101673, -0.09999,  -0.099901, -0.102508, -0.103734, -0.10431,  -0.105292,
    -0.105754, -0.104938, -0.107077, -0.109832, -0.113169, -0.116392, -0.117326, -0.115867, -0.112157, -0.108776,
    -0.106809, -0.106665, -0.107101, -0.10728,  -0.10968,  -0.110215, -0.108624, -0.107113, -0.106606, -0.104836,
    -0.103446, -0.099937, -0.098563, -0.096026, -0.096032, -0.097297, -0.098636, -0.100789, -0.100811, -0.10134,
    -0.102574, -0.104604, -0.106295, -0.109517, -0.109754, -0.106337, -0.105459, -0.103619, -0.103879, -0.104555,
    -0.104304, -0.106182, -0.1077,   -0.107955, -0.104887, -0.1028,   -0.102936, -0.104754, -0.108918, -0.109058,
    -0.109621, -0.111552, -0.11092,  -0.10979,  -0.106241, -0.099871, -0.095659, -0.093566, -0.097233, -0.103468,
    -0.107334, -0.108297, -0.109324, -0.109405, -0.106693, -0.104452, -0.104951, -0.104034, -0.103549, -0.103365,
    -0.101367, -0.10493,  -0.106295, -0.1041,   -0.102505, -0.101257, -0.102145, -0.10304};

inline constexpr std::array LSTM_M_EXPECTED_STREAMING{
    -0.0930442214012146,  -0.10166983306407928, -0.10501920431852341, -0.10657569766044617, -0.10607539117336273,
    -0.10482902824878693, -0.10231335461139679, -0.10263599455356598, -0.10327673703432083, -0.10224666446447372,
    -0.09955409169197083, -0.09567918628454208, -0.09269727766513824, -0.09150153398513794, -0.09275726974010468,
    -0.09532549232244492, -0.09780998528003693, -0.10015469789505005, -0.10317373275756836, -0.10433868318796158,
    -0.10324603319168091, -0.10241663455963135, -0.10273871570825577, -0.10367941111326218, -0.1080712229013443,
    -0.10710837692022324, -0.10402795672416687, -0.10121650993824005, -0.09804555028676987, -0.09511443227529526,
    -0.093697190284729,   -0.09358029067516327, -0.0953172892332077,  -0.09734683483839035, -0.09839528053998947,
    -0.09940513223409653, -0.10082077234983444, -0.10418295115232468, -0.10369297862052917, -0.09843398630619049,
    -0.09462916851043701, -0.09449699521064758, -0.09332452714443207, -0.0921751856803894,  -0.0941149890422821,
    -0.09765798598527908, -0.0989719033241272,  -0.10020605474710464, -0.0989457219839096,  -0.09862977266311646,
    -0.09872040152549744, -0.09861311316490173, -0.09639260917901993, -0.09633652120828629, -0.09608857333660126,
    -0.09561800211668015, -0.09596837311983109, -0.09780717641115189, -0.09911609441041946, -0.10225629806518555,
    -0.10266478359699249, -0.10359068214893341, -0.10516515374183655, -0.1063661277294159,  -0.108282171189785,
    -0.1077827513217926,  -0.1072385311126709,  -0.10627530515193939, -0.10450983792543411, -0.10326333343982697,
    -0.1000257283449173,  -0.099845752120018};

template <typename ModelT, typename T, size_t n>
void TestLstm(
    std::string model_npy,
    std::string data_npy,
    CudaEnv &cuda_env,
    std::uint32_t batch_size,
    float abs_error,
    ModelConfig config,
    const std::array<T, n> &expected) {
  if (cuda_env.DeviceCount() == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / model_npy).c_str()};

  std::unique_ptr<ModelT> model = std::make_unique<ModelT>(cuda_env, model_def, batch_size, config);
  auto const data =
      util::Tensor32::Load(test_resources_path / data_npy).template Reallocate<cex::PinnedAllocator<float>>();

  auto const num_features = model_def.NumFeatures();
  ASSERT_EQ(data.Shape().at(1), num_features);

  auto const num_steps = model->NumSteps();
  auto const init_size = (num_steps - 1) * num_features;

  auto const expected_size = expected.size();
  ASSERT_LE(expected_size, data.Shape().at(0) - num_steps + 1);
  ASSERT_EQ(expected_size % batch_size, 0);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model->InitX(initial);

  auto input = cex::MallocHost<float const *>(batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  auto const num_batches = expected_size / batch_size;
  for (std::size_t batch = 0; batch < num_batches; ++batch) {
    auto const batch_start = batch * batch_size;
    for (std::size_t i = 0; i < batch_size; ++i) {
      input[i] = data_start + (batch_start + i) * num_features;
    }

    model->PreProcess();
    model->Infer(input.get());
    model->Output(output.get());
    model->OutputWait();
    model->PostProcess();

    for (std::size_t offset = 0; offset < batch_size; ++offset) {
      EXPECT_NEAR(output[offset], expected[batch_start + offset], abs_error)
          << " batch: " << batch << "/" << num_batches << " offset: " << offset << "/" << batch_size;
    }
  }

  model = nullptr;
}

}  // namespace

TEST_F(ModelTest, InferenceLstmSFloatSingleStep) {
  ModelConfig config{};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<float>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 4e-4, config, LSTM_S_EXPECTED);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<float>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 4e-4, config, LSTM_S_EXPECTED);
}

TEST_F(ModelTest, InferenceLstmSHalfSingleStep) {
  ModelConfig config{};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<half>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 2e-3, config, LSTM_S_EXPECTED);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<half>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 2e-3, config, LSTM_S_EXPECTED);
}

TEST_F(ModelTest, InferenceLstmSFloatSingleStepStreaming) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStepStreaming};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<float>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 4e-4, config, LSTM_S_EXPECTED_STREAMING);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<float>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 4e-4, config, LSTM_S_EXPECTED_STREAMING);
}

TEST_F(ModelTest, InferenceLstmSHalfSingleStepStreaming) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStepStreaming};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<half>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 2e-3, config, LSTM_S_EXPECTED_STREAMING);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<half>>("lstm_s", "lstm_s.npy", *cuda_env_, 1, 2e-3, config, LSTM_S_EXPECTED_STREAMING);
}

TEST_F(ModelTest, InferenceLstmMFloatSingleStepTest) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStep};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<float>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 4e-4, config, LSTM_M_EXPECTED);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<float>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 4e-4, config, LSTM_M_EXPECTED);
}

TEST_F(ModelTest, InferenceLstmMHalfSingleStepTest) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStep};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<half>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 2e-3, config, LSTM_M_EXPECTED);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<half>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 2e-3, config, LSTM_M_EXPECTED);
}

TEST_F(ModelTest, InferenceLstmMFloatSingleStepStreaming) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStepStreaming};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<float>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 4e-4, config, LSTM_M_EXPECTED_STREAMING);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<float>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 4e-4, config, LSTM_M_EXPECTED_STREAMING);
}

TEST_F(ModelTest, InferenceLstmMHalfSingleStepStreaming) {
  ModelConfig config{.algorithm = ModelConfig::Algo::SingleStepStreaming};
  config.pre_compute = false;
  TestLstm<ModelSingleStep<half>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 2e-3, config, LSTM_M_EXPECTED_STREAMING);
  config.pre_compute = true;
  TestLstm<ModelSingleStep<half>>("lstm_m", "lstm_m.npy", *cuda_env_, 1, 2e-3, config, LSTM_M_EXPECTED_STREAMING);
}

#ifndef TEST_DISABLE_SINGLE_STEP_GEMV_TESTING
TEST_F(ModelAsTest, InferenceLstmSAsFloatSingleStep) {
  ModelConfig config{};
  config.single_step = true;
  config.pre_compute = false;
  contexts_ = CreateGreenContexts(LSTM_S_NUM_UNITS, 1, 0, &primary_ctx_, true);
  cuda_env_->AssignContexts(contexts_);
  config.as_supported = true;
  // TODO: Revise this test after the latest changes:
  // TestLstmS<ModelAsSingleStepGemv<float>>(*cuda_env_, 1, 4e-4, config);
}

TEST_F(ModelTest, InferenceLstmMFloatSingleStepGemv) {
  ModelConfig config{};
  config.single_step = true;
  config.pre_compute = false;
  TestLstmM<ModelSingleStepGemv<float>>(*cuda_env_, 1, 8e-4, config);
}

TEST_F(ModelTest, InferenceLstmMHalfSingleStepGemv) {
  ModelConfig config{};
  config.single_step = true;
  config.pre_compute = false;
  TestLstmM<ModelSingleStepGemv<half>>(*cuda_env_, 1, 4e-3, config);
}
#endif

namespace {
// See test/notebooks/ONNX.ipynb
std::array constexpr kExpectedC{
    -0.0197785, -0.0202075, -0.0200616, -0.0194115, -0.0185995, -0.0178914, -0.0174725, -0.0175070,
    -0.0175531, -0.0173837, -0.0171293, -0.0169771, -0.0169815, -0.0171627, -0.0171808, -0.0170669,
    -0.0171679, -0.0173488, -0.0173738, -0.0176342, -0.0180601, -0.0186779, -0.0193421, -0.0197050,
    -0.0196972, -0.0194994, -0.0191530, -0.0186313, -0.0184911, -0.0185473, -0.0185234, -0.0179954,
    -0.0172665, -0.0167573, -0.0166474, -0.0167838, -0.0171975, -0.0175385, -0.0177894, -0.0178208,
    -0.0180657, -0.0179956, -0.0179744, -0.0179217, -0.0177646, -0.0175098, -0.0174460, -0.0175836,
    -0.0176885, -0.0176318, -0.0176357, -0.0175515, -0.0175374, -0.0175626, -0.0174738, -0.0173912,
    -0.0170891, -0.0165967, -0.0161799, -0.0158031, -0.0156977, -0.0159785, -0.0165420, -0.0172218};
}  // namespace

#if 0
TEST_F(ModelTest, InferenceLstmL) {
  if (device_count_ == 0) GTEST_SKIP();

  auto const test_resources_path = std::filesystem::path(TEST_RESOURCES_DIR);
  ModelDef model_def{(test_resources_path / "lstm_l").c_str()};

  auto constexpr batch_size = 1;
  ModelConfig config{};
  config.algorithm = ModelConfig::Algo::Standard;
  Model model{*cuda_env_, model_def, batch_size, config};
  auto const data = util::Tensor32::Load(test_resources_path / "lstm_l.npy");

  auto const num_features = model_def.NumFeatures();
  ASSERT_EQ(data.Shape().at(1), num_features);

  auto const num_steps = model_def.NumSteps();
  auto const init_size = (num_steps - 1) * batch_size * num_features;

  auto constexpr &expected = kExpectedC;
  ASSERT_TRUE(expected.size() <= data.Shape().at(0) - num_steps + 1);
  util::Tensor32 initial{util::Vector32(init_size, NAN), util::ShapeVector{num_steps - 1, num_features}};
  auto const *data_ptr = data.Data().data();
  auto const *data_start = data_ptr + init_size;
  std::copy(data_ptr, data_start, initial.Data().data());
  model.InitX(initial);

  auto input = cex::MallocHost<float>(num_features * batch_size);
  auto output = cex::MallocHost<float>(batch_size);

  for (std::size_t offset = 0; offset < expected.size(); ++offset) {
    cex::Copy(input, data_start + offset * num_features, num_features, cuda_env_->Stream());
    model.Infer(input.get());
    model.InputWait();
    EXPECT_TRUE(model.InputReady());
    model.Output(output.get(), true);
    EXPECT_TRUE(model.OutputReady());
    model.OutputWait();
    EXPECT_NEAR(output[0], expected[offset], 6e-4);
  }
}

#endif
