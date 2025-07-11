cmake_minimum_required(VERSION 4.0)
cmake_policy(SET CMP0048 NEW)
if ("${CMAKE_VERSION}" VERSION_GREATER "3.12.0")
    cmake_policy(SET CMP0074 NEW)
endif ()

project(QnTools VERSION 0.0.1
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/HeavyIonAnalysis/QnTools)

set(CMAKE_CXX_STANDARD 17)

# Doxygen
option(BUILD_DOC "Build documentation" OFF)
option(QnTools_BUILD_TESTS "Build tests" ON)

# check if Doxygen is installed
if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    if(BUILD_DOC)
        find_package(Doxygen)
        if (DOXYGEN_FOUND)
            # set input and output files
            set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
            set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

            # request to configure the file
            configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
            message("Doxygen build started")

            # note the option ALL which allows to build the docs together with the application
            add_custom_target( doc_doxygen ALL
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating API documentation with Doxygen"
                    VERBATIM )
        else (DOXYGEN_FOUND)
            message("Doxygen need to be installed to generate the doxygen documentation")
        endif (DOXYGEN_FOUND)
    endif(BUILD_DOC)
endif (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")

# ROOT
find_package(ROOT 6.20 CONFIG REQUIRED)
message(STATUS "Using ROOT: ${ROOT_VERSION} <${ROOT_CONFIG}>")

set(QnTools_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(QnTools_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

IF(QnTools_BUILD_TESTS)
    include(GoogleTest)
    enable_testing()
    # Googletest
    configure_file(${QnTools_SOURCE_DIR}/tests/CMakeLists.txt.in
            ${QnTools_BINARY_DIR}/tests/googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${QnTools_BINARY_DIR}/tests/googletest-download)
    if (result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${QnTools_BINARY_DIR}/tests/googletest-download)
    if (result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif ()
    # Prevent overriding the parent project's compiler/linker settings on Windows
    #set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${QnTools_BINARY_DIR}/tests/googletest-src ${QnTools_BINARY_DIR}/tests/googletest-build EXCLUDE_FROM_ALL)
    include_directories(${gtest_SOURCE_DIR}/include)
ENDIF(QnTools_BUILD_TESTS)


# Add subdirectories
add_subdirectory(src/base)
add_subdirectory(src/correction)
add_subdirectory(src/dataframe)

# if debug enable tests
IF (QnTools_BUILD_TESTS)
    add_subdirectory(tests/base)
    add_subdirectory(tests/simulation)
    add_subdirectory(tests/correction)
ENDIF (QnTools_BUILD_TESTS)

# Install configuration
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include/QnTools)

FILE(GLOB_RECURSE headers "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
FILE(GLOB_RECURSE headers_include "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${headers_include} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Installation of ROOT dictionaries
install(FILES ${QnTools_BINARY_DIR}/src/base/libQnToolsBase_rdict.pcm DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${QnTools_BINARY_DIR}/src/base/libQnToolsBase.rootmap DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${QnTools_BINARY_DIR}/src/correction/libQnToolsCorrection_rdict.pcm DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${QnTools_BINARY_DIR}/src/correction/libQnToolsCorrection.rootmap DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS QnToolsBase QnToolsCorrection QnToolsDataFrame
        EXPORT QnToolsTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

set(INSTALL_CONFIGDIR lib/cmake/QnTools)
install(EXPORT QnToolsTargets
        FILE QnToolsTargets.cmake
        NAMESPACE QnTools::
        DESTINATION ${INSTALL_CONFIGDIR}
        )
#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/QnToolsConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/QnToolsConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/QnToolsConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR})
#Install the config, configversion
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/QnToolsConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/QnToolsConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

export(TARGETS QnToolsBase QnToolsCorrection QnToolsDataFrame NAMESPACE QnTools:: FILE QnTargets.cmake)
export(PACKAGE QnTools)


