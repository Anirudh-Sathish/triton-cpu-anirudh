add_triton_library(TritonToSCF
    LowerToSCF.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToSCFPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorDialect
)
