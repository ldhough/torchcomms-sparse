#!/bin/bash
# Download all third-party dependencies on login node
# Run this BEFORE submitting compute job

set -e

DOWNLOAD_DIR="${DOWNLOAD_DIR:-/home/egencer/scratch/torchcomms-tmp/third-party-src}"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "Downloading third-party dependencies to $DOWNLOAD_DIR..."

# fmt
echo "Downloading fmt..."
if [ ! -e "fmt" ]; then
  git clone --depth 1 -b "11.2.0" "https://github.com/fmtlib/fmt.git" fmt
fi

# zlib
echo "Downloading zlib..."
if [ ! -e "zlib" ]; then
  git clone --depth 1 -b "v1.2.13" "https://github.com/madler/zlib.git" zlib
fi

# boost (large! ~150MB+)
echo "Downloading boost (this may take a while)..."
if [ ! -e "boost" ]; then
  git clone -j 10 --recurse-submodules --depth 1 -b "boost-1.82.0" "https://github.com/boostorg/boost.git" boost
fi

# openssl
echo "Downloading openssl..."
if [ ! -e "openssl" ]; then
  git clone -j 10 --recurse-submodules --depth 1 -b "openssl-3.5.1" "https://github.com/openssl/openssl.git" openssl
fi

# xxHash
echo "Downloading xxHash..."
if [ ! -e "xxhash" ]; then
  git clone --depth 1 -b "v0.8.0" "https://github.com/Cyan4973/xxHash.git" xxhash
fi

# gflags
echo "Downloading gflags..."
if [ ! -e "gflags" ]; then
  git clone --depth 1 -b "v2.2.2" "https://github.com/gflags/gflags.git" gflags
fi

# glog
echo "Downloading glog..."
if [ ! -e "glog" ]; then
  git clone --depth 1 -b "v0.4.0" "https://github.com/google/glog.git" glog
fi

# zstd
echo "Downloading zstd..."
if [ ! -e "zstd" ]; then
  git clone --depth 1 -b "v1.5.6" "https://github.com/facebook/zstd.git" zstd
fi

# libsodium
echo "Downloading libsodium..."
if [ ! -e "sodium" ]; then
  git clone --depth 1 -b "1.0.20-RELEASE" "https://github.com/jedisct1/libsodium.git" sodium
fi

# fast_float
echo "Downloading fast_float..."
if [ ! -e "fast_float" ]; then
  git clone --depth 1 -b "v8.0.2" "https://github.com/fastfloat/fast_float.git" fast_float
fi

# libevent
echo "Downloading libevent..."
if [ ! -e "event" ]; then
  git clone --depth 1 -b "release-2.1.12-stable" "https://github.com/libevent/libevent.git" event
fi

# double-conversion
echo "Downloading double-conversion..."
if [ ! -e "double-conversion" ]; then
  git clone --depth 1 -b "v3.3.1" "https://github.com/google/double-conversion.git" double-conversion
fi

# folly and other Meta libraries (all same tag)
THIRD_PARTY_TAG="v2026.01.19.00"

echo "Downloading folly..."
if [ ! -e "folly" ]; then
  git clone --depth 1 -b "$THIRD_PARTY_TAG" "https://github.com/facebook/folly.git" folly
fi

echo "Downloading fizz..."
if [ ! -e "fizz" ]; then
  git clone --depth 1 -b "$THIRD_PARTY_TAG" "https://github.com/facebookincubator/fizz.git" fizz
fi

echo "Downloading mvfst (quic)..."
if [ ! -e "quic" ]; then
  git clone --depth 1 -b "$THIRD_PARTY_TAG" "https://github.com/facebook/mvfst" quic
fi

echo "Downloading wangle..."
if [ ! -e "wangle" ]; then
  git clone --depth 1 -b "$THIRD_PARTY_TAG" "https://github.com/facebook/wangle.git" wangle
fi

echo "Downloading fbthrift..."
if [ ! -e "thrift" ]; then
  git clone --depth 1 -b "$THIRD_PARTY_TAG" "https://github.com/facebook/fbthrift.git" thrift
fi

echo ""
echo "======================================"
echo "All dependencies downloaded to:"
echo "$DOWNLOAD_DIR"
echo "======================================"
echo ""
echo "Directory contents:"
ls -la "$DOWNLOAD_DIR"
