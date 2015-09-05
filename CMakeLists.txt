project(vasr)

cmake_minimum_required(VERSION 2.8)


find_package(OpenCV)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


################################################################################
## Global compilation parameters                                              ##
################################################################################


##
## Target platform selection
##
set(CMAKE_BUILD_PLATFORM "Auto"  CACHE STRING  "Target platform selection")
set_property(CACHE  CMAKE_BUILD_PLATFORM PROPERTY STRINGS "Auto" "64 bits" "32 bits" )

##
## Compilation mode
##
set(CMAKE_BUILD_TYPE "Auto"  CACHE STRING  "Compilation mode")
set_property(CACHE  CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")


################################################################################
## Handle automatic parameters generation                                     ##
################################################################################


##
## Operating system identification
##
set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")


##
## If omited, the default compilation mode is "Release"
##
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "No build type selected, default to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif()


##
## Select the target platform if the user chose "Auto"
##
if(${CMAKE_BUILD_PLATFORM} STREQUAL "Auto")
    if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
        set(CMAKE_BUILD_PLATFORM "32 bits")
    elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set(CMAKE_BUILD_PLATFORM "64 bits")
    endif()
endif()


##
## Define variables for additional libraries
##
find_package (Threads)
set(VASR_LIBS_THREAD ${CMAKE_THREAD_LIBS_INIT})
set(VASR_LIBS_DL ${CMAKE_DL_LIBS})


##
## Define target platform options
##
if (UNIX)
    set(__ADNDROID__ 0)
    if(CMAKE_BUILD_PLATFORM STREQUAL "32 bits")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32" )
    elseif(CMAKE_BUILD_PLATFORM STREQUAL "64 bits")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m64" )
    endif()
endif()


##
## Define GCC/C-Lang options (Unix only)
##
if (UNIX)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC -std=gnu99 -Wall -Werror=implicit-function-declaration")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror=implicit-function-declaration")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -s -O3")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s -O3")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    # Create the __FILENAME__ macro to give a reduced source file name (debug purposes)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
endif()

##
## Define flags de compilação exclusivas de ambientes Windows (usando Visual Studio)
##
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()



file(GLOB PLS_SOURCES "source/*.cpp")

include_directories("include")
add_executable(pls ${PLS_SOURCES})
target_link_libraries(pls ${OpenCV_LIBS})



message("")
message("  ======== Enviroment ========")
message("                Host:  ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message("               CMake:  ${CMAKE_VERSION}")
message("           Generator:  ${CMAKE_GENERATOR}")
message("          Build tool:  ${CMAKE_BUILD_TOOL}")
if(MSVC)
   message("        MSVC version:  ${MSVC_VERSION}")
endif()


message("")
message("  ========== Target ==========")
message("    Operating system:  ${CMAKE_SYSTEM_NAME}")
message("            Platform:  ${CMAKE_BUILD_PLATFORM}")
if (ENABLE_ANDROID)
    message("           API Level:  ${ANDROID_API_LEVEL}")
    message("       NDK directory:  ${ANDROID_DIR}")
endif()
message("          Build type:  ${CMAKE_BUILD_TYPE}")


message("")
message("  ======== Compilation =======")
message("      Root directory: ${ROOT_DIRECTORY}")

if (CMAKE_CXX_COMPILER_VERSION)
    set(CUSTOM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_COMPILER_IS_CLANGCXX)
    set(CUSTOM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CLANG_REGEX_VERSION})")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CUSTOM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_GCC_REGEX_VERSION})")
else()
    set(CUSTOM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${CUSTOM_COMPILER_STR}" CUSTOM_COMPILER_STR)
message("        C++ Compiler:  ${CUSTOM_COMPILER_STR}")
message(" C++ flags (Release):  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message("   C++ flags (Debug):   ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message("          C Compiler:  ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}")
message("   C flags (Release):  ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
message("     C flags (Debug):  ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
if(WIN32)
    message("  LD flags (Release): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    message("    LD flags (Debug):   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
    message("  LD flags (Release): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    message("    LD flags (Debug):   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()
message("")
