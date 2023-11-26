#pragma once
#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>


// To calculate a color id
int inline adler32(const unsigned char* data) {
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;
    int colorId;
    size_t index;
    for (index = 0; data[index] != 0; ++index) {
        a = (a + data[index] * 2) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    colorId = (b << 16) | a;

    int red, green, blue;
    red = colorId & 0x000000ff;
    green = (colorId & 0x000ff000) >> 12;
    blue = (colorId & 0x0ff00000) >> 20;
    if (red < 64 & green < 64 & blue < 64) {
        red = red * 3;
        green = green * 3 + 64;
        blue = blue * 4;
    }

    return 0xff000000 | (red << 16) | (green << 8) | (blue);
}

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
