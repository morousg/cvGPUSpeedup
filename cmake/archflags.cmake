function (add_msvc_flags TARGET_NAME)
    # Add MSVC specific flags for the target
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
        SET(ARCH_FLAGS "AVX2" CACHE STRING "instrucion set to use")
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS AVX AVX2 AVX512 AVX10.1 disabled)  
        option(ARCH_FLAGS "CPU arch" "AVX2")
            
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)        
        endif()
        
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ARM64")
        #we don't have  windows arm64 hw to test 
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS armv8.2 disabled)  
        option(ARCH_FLAGS "CPU arch" "AVX2")
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)        
        endif()
    endif()
endfunction()

#works both for x86_64 and aarch64      
    #we currently don't test avx10.1 with gcc11 on ubuntu 22.04
    #armv9-a (GH100)  not available in gcc11
    #for arm64 we test on jetson orin and grace-hopper
    #sandybridge: minimum avx arch | haswell: minimum avx2 arch | skylake: minium avx512 arch
    #AVX10.1 only gcc 14 (and >15 for diamondrapids)
    
function (add_unix_flags TARGET_NAME)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native sandybridge haswell skylake-avx512 diamondrapids disabled)             
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")         
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native armv8.2-a armv9-a disabled)  
    endif()
    
    option(ARCH_FLAGS "native" "instrucion set to use")          
    if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${ARCH_FLAGS} -mtune=${ARCH_FLAGS}>)     
    endif()                        

endfunction()
# Add architecture-specific optimization flags for the target
function(add_optimization_flags TARGET_NAME)    
    if (MSVC)        
        add_msvc_flags(${TARGET_NAME})
    elseif (UNIX)             
        add_unix_flags(${TARGET_NAME})    
    endif()
endfunction()