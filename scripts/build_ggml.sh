#!/usr/bin/env bash
#
# Build ggml as static libraries that the KittensRnD Xcode project links
# against. Run this once after cloning (or after pulling a new
# vendors/llama.cpp commit) before opening the .xcodeproj — Xcode does
# not invoke cmake itself.
#
# Builds three slices: macOS arm64 (CPU+Accelerate), iphoneos arm64,
# iphonesimulator arm64. We deliberately skip ggml-metal: the model is
# dominated by small LSTM ops and the per-op kernel-launch cost on
# Metal exceeds the actual compute. CPU runs ~9.3x realtime on M-series.
#
# Usage:
#   ./scripts/build_ggml.sh              # all three slices
#   ./scripts/build_ggml.sh macos        # just macOS
#   ./scripts/build_ggml.sh ios          # just both iOS slices

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/vendors/llama.cpp"

if [ ! -d "$LLAMA" ] || [ -z "$(ls -A "$LLAMA" 2>/dev/null)" ]; then
    echo "vendors/llama.cpp is empty — run: git submodule update --init"
    exit 1
fi

build_macos() {
    local BUILD="$LLAMA/build-cpu"
    rm -rf "$BUILD"
    # Pin the deployment target to match KittensRnD.xcodeproj's
    # MACOSX_DEPLOYMENT_TARGET — without it CMake picks the host SDK
    # default (e.g. 26.0) and the linker warns "object file was built
    # for newer 'macOS' version than being linked".
    cmake -S "$LLAMA" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple
    cmake --build "$BUILD" --config Release -j \
        --target ggml ggml-cpu ggml-base ggml-blas
    echo "macOS arm64 libs in $BUILD/ggml/src/"
}

# iOS uses the Xcode generator so one cmake config produces both
# Release-iphoneos/ and Release-iphonesimulator/ subdirs after we
# build twice — once for each SDK.
build_ios() {
    local BUILD="$LLAMA/build-ios"
    rm -rf "$BUILD"
    # Bump to iOS 18.0 to match KittensRnD.xcodeproj (CoreML's
    # MLModelConfiguration.functionName is iOS 18+). Same warning to
    # avoid as in build_macos().
    cmake -G Xcode -S "$LLAMA" -B "$BUILD" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 \
        -DCMAKE_OSX_SYSROOT=iphoneos \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple
    cmake --build "$BUILD" --config Release \
        --target ggml ggml-cpu ggml-base ggml-blas -- -sdk iphoneos
    cmake --build "$BUILD" --config Release \
        --target ggml ggml-cpu ggml-base ggml-blas -- -sdk iphonesimulator
    echo "iOS arm64 libs in $BUILD/ggml/src/Release-{iphoneos,iphonesimulator}/"
}

case "${1:-all}" in
    macos)  build_macos ;;
    ios)    build_ios ;;
    all)    build_macos; build_ios ;;
    *)      echo "usage: $0 [macos|ios|all]"; exit 1 ;;
esac
