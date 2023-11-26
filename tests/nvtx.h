#pragma once
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#ifdef WIN32
// nvtx headers include Windows.h at some point, and Windows.h defines ERROR 0
// this breaks the compilation of Ceres

#undef ERROR
#endif
#define PUSH_RANGE_PAYLOAD(name, value)                                      \
    {                                                                        \
        double v = (double)value;                                            \
        int colorId = adler32(reinterpret_cast<const unsigned char*>(name)); \
        nvtxEventAttributes_t eventAttrib = {0};                             \
        eventAttrib.version = NVTX_VERSION;                                  \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                    \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                             \
        eventAttrib.color = colorId;                                         \
        eventAttrib.payload.llValue = v;                                     \
        eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;                  \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                   \
        eventAttrib.message.ascii = name;                                    \
        nvtxRangePushEx(&eventAttrib);                                       \
    }

#define PUSH_RANGE(name)                                                     \
    {                                                                        \
        int colorId = adler32(reinterpret_cast<const unsigned char*>(name)); \
        nvtxEventAttributes_t eventAttrib = {0};                             \
        eventAttrib.version = NVTX_VERSION;                                  \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                    \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                             \
        eventAttrib.color = colorId;                                         \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                   \
        eventAttrib.message.ascii = name;                                    \
        nvtxRangePushEx(&eventAttrib);                                       \
    }
#define POP_RANGE nvtxRangePop();

#define CUDA_MARK(name)                                    \
    {                                                      \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxMarkEx(&eventAttrib);                          \
    }

#else
#define PUSH_RANGE_PAYLOAD(name, value)
#define PUSH_RANGE(name)
#define POP_RANGE
#define CUDA_MARK(name)
#endif


class PUSH_RANGE_RAII {
public:
    explicit PUSH_RANGE_RAII(const char* name) {
        PUSH_RANGE(name);
    }

    ~PUSH_RANGE_RAII() {
        POP_RANGE;
    }
};
