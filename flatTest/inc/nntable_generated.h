// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_NNTABLE_NNEXECUTOR_H_
#define FLATBUFFERS_GENERATED_NNTABLE_NNEXECUTOR_H_

#include "flatbuffers/flatbuffers.h"

namespace NNExecutor {

struct Kernel;

struct CompOutput;

struct Conv;

struct Relu;

enum kernel_info {
  kernel_info_NONE = 0,
  kernel_info_Conv = 1,
  kernel_info_Relu = 2,
  kernel_info_MIN = kernel_info_NONE,
  kernel_info_MAX = kernel_info_Relu
};

inline const kernel_info (&EnumValueskernel_info())[3] {
  static const kernel_info values[] = {
    kernel_info_NONE,
    kernel_info_Conv,
    kernel_info_Relu
  };
  return values;
}

inline const char * const *EnumNameskernel_info() {
  static const char * const names[] = {
    "NONE",
    "Conv",
    "Relu",
    nullptr
  };
  return names;
}

inline const char *EnumNamekernel_info(kernel_info e) {
  if (e < kernel_info_NONE || e > kernel_info_Relu) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNameskernel_info()[index];
}

template<typename T> struct kernel_infoTraits {
  static const kernel_info enum_value = kernel_info_NONE;
};

template<> struct kernel_infoTraits<Conv> {
  static const kernel_info enum_value = kernel_info_Conv;
};

template<> struct kernel_infoTraits<Relu> {
  static const kernel_info enum_value = kernel_info_Relu;
};

bool Verifykernel_info(flatbuffers::Verifier &verifier, const void *obj, kernel_info type);
bool Verifykernel_infoVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types);

struct Kernel FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_OPCODE = 4,
    VT_OPINFO_TYPE = 6,
    VT_OPINFO = 8
  };
  const flatbuffers::String *opcode() const {
    return GetPointer<const flatbuffers::String *>(VT_OPCODE);
  }
  kernel_info opinfo_type() const {
    return static_cast<kernel_info>(GetField<uint8_t>(VT_OPINFO_TYPE, 0));
  }
  const void *opinfo() const {
    return GetPointer<const void *>(VT_OPINFO);
  }
  template<typename T> const T *opinfo_as() const;
  const Conv *opinfo_as_Conv() const {
    return opinfo_type() == kernel_info_Conv ? static_cast<const Conv *>(opinfo()) : nullptr;
  }
  const Relu *opinfo_as_Relu() const {
    return opinfo_type() == kernel_info_Relu ? static_cast<const Relu *>(opinfo()) : nullptr;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_OPCODE) &&
           verifier.VerifyString(opcode()) &&
           VerifyField<uint8_t>(verifier, VT_OPINFO_TYPE) &&
           VerifyOffset(verifier, VT_OPINFO) &&
           Verifykernel_info(verifier, opinfo(), opinfo_type()) &&
           verifier.EndTable();
  }
};

template<> inline const Conv *Kernel::opinfo_as<Conv>() const {
  return opinfo_as_Conv();
}

template<> inline const Relu *Kernel::opinfo_as<Relu>() const {
  return opinfo_as_Relu();
}

struct KernelBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_opcode(flatbuffers::Offset<flatbuffers::String> opcode) {
    fbb_.AddOffset(Kernel::VT_OPCODE, opcode);
  }
  void add_opinfo_type(kernel_info opinfo_type) {
    fbb_.AddElement<uint8_t>(Kernel::VT_OPINFO_TYPE, static_cast<uint8_t>(opinfo_type), 0);
  }
  void add_opinfo(flatbuffers::Offset<void> opinfo) {
    fbb_.AddOffset(Kernel::VT_OPINFO, opinfo);
  }
  explicit KernelBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  KernelBuilder &operator=(const KernelBuilder &);
  flatbuffers::Offset<Kernel> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Kernel>(end);
    return o;
  }
};

inline flatbuffers::Offset<Kernel> CreateKernel(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> opcode = 0,
    kernel_info opinfo_type = kernel_info_NONE,
    flatbuffers::Offset<void> opinfo = 0) {
  KernelBuilder builder_(_fbb);
  builder_.add_opinfo(opinfo);
  builder_.add_opcode(opcode);
  builder_.add_opinfo_type(opinfo_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<Kernel> CreateKernelDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *opcode = nullptr,
    kernel_info opinfo_type = kernel_info_NONE,
    flatbuffers::Offset<void> opinfo = 0) {
  auto opcode__ = opcode ? _fbb.CreateString(opcode) : 0;
  return NNExecutor::CreateKernel(
      _fbb,
      opcode__,
      opinfo_type,
      opinfo);
}

struct CompOutput FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_INST = 4
  };
  const flatbuffers::Vector<flatbuffers::Offset<Kernel>> *inst() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Kernel>> *>(VT_INST);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_INST) &&
           verifier.VerifyVector(inst()) &&
           verifier.VerifyVectorOfTables(inst()) &&
           verifier.EndTable();
  }
};

struct CompOutputBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_inst(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Kernel>>> inst) {
    fbb_.AddOffset(CompOutput::VT_INST, inst);
  }
  explicit CompOutputBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  CompOutputBuilder &operator=(const CompOutputBuilder &);
  flatbuffers::Offset<CompOutput> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<CompOutput>(end);
    return o;
  }
};

inline flatbuffers::Offset<CompOutput> CreateCompOutput(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Kernel>>> inst = 0) {
  CompOutputBuilder builder_(_fbb);
  builder_.add_inst(inst);
  return builder_.Finish();
}

inline flatbuffers::Offset<CompOutput> CreateCompOutputDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<Kernel>> *inst = nullptr) {
  auto inst__ = inst ? _fbb.CreateVector<flatbuffers::Offset<Kernel>>(*inst) : 0;
  return NNExecutor::CreateCompOutput(
      _fbb,
      inst__);
}

struct Conv FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_KERNEL_SIZE = 6,
    VT_STRIDE_SIZE = 8,
    VT_PAD_SIZE = 10,
    VT_WEIGHT = 12
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  int32_t kernel_size() const {
    return GetField<int32_t>(VT_KERNEL_SIZE, 0);
  }
  int32_t stride_size() const {
    return GetField<int32_t>(VT_STRIDE_SIZE, 0);
  }
  int32_t pad_size() const {
    return GetField<int32_t>(VT_PAD_SIZE, 0);
  }
  const flatbuffers::Vector<float> *weight() const {
    return GetPointer<const flatbuffers::Vector<float> *>(VT_WEIGHT);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<int32_t>(verifier, VT_KERNEL_SIZE) &&
           VerifyField<int32_t>(verifier, VT_STRIDE_SIZE) &&
           VerifyField<int32_t>(verifier, VT_PAD_SIZE) &&
           VerifyOffset(verifier, VT_WEIGHT) &&
           verifier.VerifyVector(weight()) &&
           verifier.EndTable();
  }
};

struct ConvBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(Conv::VT_NAME, name);
  }
  void add_kernel_size(int32_t kernel_size) {
    fbb_.AddElement<int32_t>(Conv::VT_KERNEL_SIZE, kernel_size, 0);
  }
  void add_stride_size(int32_t stride_size) {
    fbb_.AddElement<int32_t>(Conv::VT_STRIDE_SIZE, stride_size, 0);
  }
  void add_pad_size(int32_t pad_size) {
    fbb_.AddElement<int32_t>(Conv::VT_PAD_SIZE, pad_size, 0);
  }
  void add_weight(flatbuffers::Offset<flatbuffers::Vector<float>> weight) {
    fbb_.AddOffset(Conv::VT_WEIGHT, weight);
  }
  explicit ConvBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ConvBuilder &operator=(const ConvBuilder &);
  flatbuffers::Offset<Conv> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Conv>(end);
    return o;
  }
};

inline flatbuffers::Offset<Conv> CreateConv(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    int32_t kernel_size = 0,
    int32_t stride_size = 0,
    int32_t pad_size = 0,
    flatbuffers::Offset<flatbuffers::Vector<float>> weight = 0) {
  ConvBuilder builder_(_fbb);
  builder_.add_weight(weight);
  builder_.add_pad_size(pad_size);
  builder_.add_stride_size(stride_size);
  builder_.add_kernel_size(kernel_size);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<Conv> CreateConvDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    int32_t kernel_size = 0,
    int32_t stride_size = 0,
    int32_t pad_size = 0,
    const std::vector<float> *weight = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto weight__ = weight ? _fbb.CreateVector<float>(*weight) : 0;
  return NNExecutor::CreateConv(
      _fbb,
      name__,
      kernel_size,
      stride_size,
      pad_size,
      weight__);
}

struct Relu FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_TYPE = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::String *type() const {
    return GetPointer<const flatbuffers::String *>(VT_TYPE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_TYPE) &&
           verifier.VerifyString(type()) &&
           verifier.EndTable();
  }
};

struct ReluBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(Relu::VT_NAME, name);
  }
  void add_type(flatbuffers::Offset<flatbuffers::String> type) {
    fbb_.AddOffset(Relu::VT_TYPE, type);
  }
  explicit ReluBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ReluBuilder &operator=(const ReluBuilder &);
  flatbuffers::Offset<Relu> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Relu>(end);
    return o;
  }
};

inline flatbuffers::Offset<Relu> CreateRelu(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::String> type = 0) {
  ReluBuilder builder_(_fbb);
  builder_.add_type(type);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<Relu> CreateReluDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const char *type = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto type__ = type ? _fbb.CreateString(type) : 0;
  return NNExecutor::CreateRelu(
      _fbb,
      name__,
      type__);
}

inline bool Verifykernel_info(flatbuffers::Verifier &verifier, const void *obj, kernel_info type) {
  switch (type) {
    case kernel_info_NONE: {
      return true;
    }
    case kernel_info_Conv: {
      auto ptr = reinterpret_cast<const Conv *>(obj);
      return verifier.VerifyTable(ptr);
    }
    case kernel_info_Relu: {
      auto ptr = reinterpret_cast<const Relu *>(obj);
      return verifier.VerifyTable(ptr);
    }
    default: return false;
  }
}

inline bool Verifykernel_infoVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types) {
  if (!values || !types) return !values && !types;
  if (values->size() != types->size()) return false;
  for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
    if (!Verifykernel_info(
        verifier,  values->Get(i), types->GetEnum<kernel_info>(i))) {
      return false;
    }
  }
  return true;
}

inline const NNExecutor::CompOutput *GetCompOutput(const void *buf) {
  return flatbuffers::GetRoot<NNExecutor::CompOutput>(buf);
}

inline const NNExecutor::CompOutput *GetSizePrefixedCompOutput(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<NNExecutor::CompOutput>(buf);
}

inline bool VerifyCompOutputBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<NNExecutor::CompOutput>(nullptr);
}

inline bool VerifySizePrefixedCompOutputBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<NNExecutor::CompOutput>(nullptr);
}

inline void FinishCompOutputBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<NNExecutor::CompOutput> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedCompOutputBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<NNExecutor::CompOutput> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace NNExecutor

#endif  // FLATBUFFERS_GENERATED_NNTABLE_NNEXECUTOR_H_
