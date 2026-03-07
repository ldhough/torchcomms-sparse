#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -A bhatele-lab-cmsc
#SBATCH -t 2-00:00:00
#SBATCH --mem=64g
#SBATCH --mem-bind=local

set -x

# Path to pre-downloaded dependencies (set by download_deps_login.sh)
THIRD_PARTY_SRC="${THIRD_PARTY_SRC:-/home/egencer/scratch/torchcomms-tmp/third-party-src}"

# Number of parallel build jobs (defaults to all available cores)
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
echo "Using $BUILD_JOBS parallel build jobs"

# Zaratan module loads
module load gcc/11.3.0
module load cmake/3.26.3
module load cuda/12.3.0/gcc/11.3.0/zen2
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load boost/1.82.0/gcc/11.3.0/zen2
module load nccl/2.18.1-1/gcc/11.3.0/zen2
module load zlib/1.2.13/gcc/11.3.0/zen2

# Unset NCCL_HOME set by nccl module, then set to local NCCLX source
unset NCCL_HOME
export NCCL_HOME="${PWD}/comms/ncclx/stable"

# Activate Python virtual environment
VENV_PATH="${VENV_PATH:-/home/egencer/scratch/torchcomms-venv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
  echo "Activated virtual environment: ${VENV_PATH}"
else
  echo "WARNING: Virtual environment not found at ${VENV_PATH}"
  echo "Create it with: python3 -m venv ${VENV_PATH}"
fi

# Hardcoded install prefix for Zaratan
INSTALL_PREFIX="/home/egencer/scratch/torchcomms-tmp"
export INSTALL_PREFIX
mkdir -p "${INSTALL_PREFIX}"

function do_cmake_build() {
  local source_dir="$1"
  local extra_flags="$2"
  # Clear stale cmake cache so the shared build-output dir can be
  # reconfigured for a different source tree.
  rm -f CMakeCache.txt
  rm -rf CMakeFiles
  cmake -G Ninja \
    -DCMAKE_C_COMPILER="${CC_FOR_BUILD:-$(command -v gcc)}" \
    -DCMAKE_CXX_COMPILER="${CXX_FOR_BUILD:-$(command -v g++)}" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_MODULE_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_DIR="$INSTALL_PREFIX" \
    -DBIN_INSTALL_DIR="$INSTALL_PREFIX/bin" \
    -DLIB_INSTALL_DIR="$INSTALL_PREFIX/$LIB_SUFFIX" \
    -DINCLUDE_INSTALL_DIR="$INSTALL_PREFIX/include" \
    -DCMAKE_INSTALL_INCLUDEDIR="$INSTALL_PREFIX/include" \
    -DCMAKE_INSTALL_LIBDIR="$INSTALL_PREFIX/$LIB_SUFFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.22 \
    $extra_flags \
    -S "${source_dir}"
  ninja -j${BUILD_JOBS}
  ninja -j${BUILD_JOBS} install
}

function clean_third_party {
  local library_name="$1"
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -rf "${INSTALL_PREFIX}"/include/"${library_name}"*/
    rm -rf "${INSTALL_PREFIX}"/include/"${library_name}"*.h
  fi
}

function build_fb_oss_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  # Determine source directory
  local source_dir
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    source_dir="${THIRD_PARTY_SRC}/${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
    source_dir="${PWD}/${library_name}"
  else
    source_dir="${PWD}/${library_name}"
  fi
  
  # Determine actual CMake source path
  local cmake_source="${source_dir}"
  if [ -f "${source_dir}/${library_name}/CMakeLists.txt" ]; then
    cmake_source="${source_dir}/${library_name}"
  elif [ -f "${source_dir}/build/cmake/CMakeLists.txt" ]; then
    cmake_source="${source_dir}/build/cmake"
  elif [ -f "${source_dir}/cmake_unofficial/CMakeLists.txt" ]; then
    cmake_source="${source_dir}/cmake_unofficial"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  
  # Use build directory in /tmp to avoid filesystem issues
  local build_dir="/tmp/build-${library_name}-$$"
  if [[ "${CLEAN_THIRD_PARTY}" == 1 || "${INCREMENTAL_BUILD}" == 0 ]]; then
    rm -rf "${build_dir}"
  fi
  mkdir -p "${build_dir}"
  pushd "${build_dir}"
  do_cmake_build "$cmake_source" "$extra_flags"
  popd
}

function build_automake_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  # Determine source directory
  local source_dir
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    source_dir="${THIRD_PARTY_SRC}/${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
    source_dir="${PWD}/${library_name}"
  else
    source_dir="${PWD}/${library_name}"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$source_dir"
  ./configure --prefix="$INSTALL_PREFIX" --disable-pie

  make -j${BUILD_JOBS}
  make install
  popd
}

function build_boost() {
  local repo_url="https://github.com/boostorg/boost.git"
  local repo_tag="boost-1.82.0"
  local library_name="boost"
  local extra_flags=""

  # clean up existing boost
  clean_third_party "$library_name"

  # Determine source directory
  local source_dir
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    source_dir="${THIRD_PARTY_SRC}/${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
    source_dir="${PWD}/${library_name}"
  else
    source_dir="${PWD}/${library_name}"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$source_dir"
  ./bootstrap.sh --prefix="$INSTALL_PREFIX" --libdir="$INSTALL_PREFIX/$LIB_SUFFIX" --without-libraries=python
  ./b2 -q cxxflags=-fPIC cflags=-fPIC install
  popd
}

function build_openssl() {
  local repo_url="https://github.com/openssl/openssl.git"
  local repo_tag="openssl-3.5.1"
  local library_name="openssl"
  local extra_flags=""

  # clean up existing openssl
  clean_third_party "$library_name"

  # Determine source directory
  local source_dir
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    source_dir="${THIRD_PARTY_SRC}/${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
    source_dir="${PWD}/${library_name}"
  else
    source_dir="${PWD}/${library_name}"
  fi

  pushd "$source_dir"
  ./config no-shared --prefix="$INSTALL_PREFIX" --openssldir="$INSTALL_PREFIX" --libdir=lib

  make -j${BUILD_JOBS}
  make install
  popd
}

function build_third_party {
  # build third-party libraries
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -f "${INSTALL_PREFIX}"/*.cmake 2>/dev/null || true
  fi
  local third_party_tag="v2026.01.19.00"

  mkdir -p /tmp/third-party
  pushd /tmp/third-party
  build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF"
  build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON"
  build_fb_oss_library "https://github.com/madler/zlib.git" "v1.2.13" zlib "-DZLIB_BUILD_TESTING=OFF"
  build_boost
  build_openssl
  build_fb_oss_library "https://github.com/Cyan4973/xxHash.git" "v0.8.0" xxhash
  # we need both static and dynamic gflags since thrift generator can't
  # statically link against glog.
  build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags
  build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags "-DBUILD_SHARED_LIBS=ON"
  # we need both static and dynamic glog since thrift generator can't
  # statically link against glog.
  build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog
  build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog "-DBUILD_SHARED_LIBS=ON"
  build_fb_oss_library "https://github.com/facebook/zstd.git" "v1.5.6" zstd
  build_automake_library "https://github.com/jedisct1/libsodium.git" "1.0.20-RELEASE" sodium
  build_fb_oss_library "https://github.com/fastfloat/fast_float.git" "v8.0.2" fast_float "-DFASTFLOAT_INSTALL=ON"
  build_fb_oss_library "https://github.com/libevent/libevent.git" "release-2.1.12-stable" event "-DEVENT__DISABLE_TESTS=ON -DEVENT__DISABLE_BENCHMARK=ON -DEVENT__DISABLE_SAMPLES=ON"
  build_fb_oss_library "https://github.com/google/double-conversion.git" "v3.3.1" double-conversion
  build_fb_oss_library "https://github.com/facebook/folly.git" "$third_party_tag" folly "-DUSE_STATIC_DEPS_ON_UNIX=ON -DOPENSSL_USE_STATIC_LIBS=ON"

  # TODO: migrate out all dependencies for feedstock
  if [[ -z "${NCCL_FEEDSTOCK_BUILD}" ]]; then
    build_fb_oss_library "https://github.com/facebookincubator/fizz.git" "$third_party_tag" fizz "-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF"
    build_fb_oss_library "https://github.com/facebook/mvfst" "$third_party_tag" quic
    build_fb_oss_library "https://github.com/facebook/wangle.git" "$third_party_tag" wangle "-DBUILD_TESTS=OFF"
  fi
  build_fb_oss_library "https://github.com/facebook/fbthrift.git" "$third_party_tag" thrift
  popd
}

function build_comms_tracing_service {
  local include_prefix="comms/analyzer/if"
  local base_dir="${PWD}"
  local build_dir=/tmp/build/comms_tracing_service

  mkdir -p "$build_dir"
  pushd "$build_dir"
  # set up the directory structure
  mkdir -p "$include_prefix"
  cp -r "${base_dir}/${include_prefix}"/* "$include_prefix"
  mv "$include_prefix"/CMakeLists.txt .

  # set up the build config
  cp -r /tmp/third-party/thrift/build .
  # Generate a minimal FBThriftConfig.cmake that points at the install prefix.
  # The installed version uses PACKAGE_PREFIX_DIR relative to its own location,
  # which breaks when copied elsewhere.
  cat > build/fbcode_builder/CMake/FBThriftConfig.cmake <<FBCFG
set(FBTHRIFT_INCLUDE_DIR "${INSTALL_PREFIX}/include")
set(FBTHRIFT_COMPILER "${INSTALL_PREFIX}/bin/thrift1")
find_package(Xxhash REQUIRED)
find_package(ZLIB REQUIRED)
find_package(mvfst CONFIG REQUIRED)
if (NOT TARGET FBThrift::thriftcpp2)
  include("${INSTALL_PREFIX}/FBThriftTargets.cmake")
endif()
set(FBThrift_FOUND True)
FBCFG

  # build the thrift service library
  cd build
  do_cmake_build ..

  popd
}

if [ -z "$DEV_SIGNATURE" ]; then
    is_git=$(git rev-parse --is-inside-work-tree)
    if [ $is_git ]; then
        DEV_SIGNATURE="git-"$(git rev-parse --short HEAD)
    else
        echo "Cannot detect source repository hash. Skip"
        DEV_SIGNATURE=""
    fi
fi

set -e
export LIB_PREFIX="lib64"

BUILDDIR=${BUILDDIR:="${PWD}/build/ncclx"}
CUDA_HOME=${CUDA_HOME:="/usr/local/cuda"}
# Auto-detect CUDA_HOME from nvcc if the default doesn't exist
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  _nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$_nvcc_path" ]]; then
    CUDA_HOME="$(dirname "$(dirname "$_nvcc_path")")"
  fi
  unset _nvcc_path
fi
NVCC_ARCH=${NVCC_ARCH:="a100"}

# Add b200 support if CUDA 12.8+ is available
CUDA_VERSION=$("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
if [[ "$CUDA_MAJOR" -gt 12 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
    NVCC_ARCH="${NVCC_ARCH},b200"
fi
NCCL_FP8=${NCCL_FP8:=1}
CLEAN_BUILD=${CLEAN_BUILD:=0}
INCREMENTAL_BUILD=${INCREMENTAL_BUILD:=1}

# Validate install prefix
if [[ -z "${INSTALL_PREFIX}" || "${INSTALL_PREFIX}" == "/" ]]; then
  echo "ERROR: INSTALL_PREFIX is empty or '/'."
  exit 1
fi
mkdir -p "${INSTALL_PREFIX}"

# Build CMAKE_PREFIX_PATH: our install prefix first, then anything the
# environment (or modules) already added.
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH}"
else
  export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
fi

if [[ -z "${LIB_SUFFIX:-}" ]]; then
  if [[ -d "${INSTALL_PREFIX}/lib64" ]]; then
    LIB_SUFFIX=lib64
  else
    LIB_SUFFIX=lib
  fi
fi
CONDA_INCLUDE_DIR="${INSTALL_PREFIX}/include"
CONDA_LIB_DIR="${INSTALL_PREFIX}/${LIB_SUFFIX}"
NCCL_HOOK_LIBS=${NCCL_HOOK_LIBS:=0}
NCCL_HOME=${NCCL_HOME:="${PWD}/comms/ncclx/stable"}
BASE_DIR=${BASE_DIR:="${PWD}"}
CUDARTLIB=cudart_static
THIRD_PARTY_LDFLAGS=""

if [[ -z "${NCCL_BUILD_SKIP_DEPS}" ]]; then
  echo "Building dependencies"
  build_third_party
fi
if [[ -z "${NCCL_BUILD_SKIP_DEPS}" || -n "${NCCL_BUILD_TRACING_SERVICE}" ]]; then
  build_comms_tracing_service
fi

# Generate nccl_cvars files (these are no longer checked into the repo)
# The files are generated by extractcvars.py which reads nccl_cvars.yaml and nccl_cvars.cc.in
echo "Generating nccl_cvars files..."
CVARS_DIR="$BASE_DIR/comms/utils/cvars"

# Validate that the required source files exist
if [ ! -f "$CVARS_DIR/extractcvars.py" ]; then
  echo "ERROR: extractcvars.py not found at $CVARS_DIR/extractcvars.py"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.yaml" ]; then
  echo "ERROR: nccl_cvars.yaml not found at $CVARS_DIR/nccl_cvars.yaml"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.cc.in" ]; then
  echo "ERROR: nccl_cvars.cc.in not found at $CVARS_DIR/nccl_cvars.cc.in"
  exit 1
fi

# Ensure PyYAML is available for extractcvars.py (no conda requirement).
python3 - <<'PY'
try:
  import yaml  # noqa: F401
except Exception as e:
  raise SystemExit("PyYAML not found. Install it in your Python env (e.g., `python3 -m pip install pyyaml`).")
PY

# Run the extractcvars.py script directly to generate the files
export NCCL_CVARS_OUTPUT_DIR="$CVARS_DIR"
python3 "$CVARS_DIR/extractcvars.py"

# Verify the files were generated
if [ ! -f "$CVARS_DIR/nccl_cvars.h" ] || [ ! -f "$CVARS_DIR/nccl_cvars.cc" ]; then
  echo "ERROR: Failed to generate nccl_cvars files"
  exit 1
fi
echo "Successfully generated nccl_cvars files in $CVARS_DIR"

# set up the third-party ldflags
export PKG_CONFIG_PATH="${CONDA_LIB_DIR}"/pkgconfig
THRIFT_SERVICE_LDFLAGS=(
  "-l:libcomms_tracing_service.a"
  "-Wl,--start-group"
  "-l:libasync.a"
  "-l:libconcurrency.a"
  "-l:libthrift-core.a"
  "-l:libthriftanyrep.a"
  "-l:libthriftcpp2.a"
  "-l:libthriftmetadata.a"
  "-l:libthriftprotocol.a"
  "-l:libthrifttype.a"
  "-l:libthrifttyperep.a"
  "-l:librpcmetadata.a"
  "-l:libruntime.a"
  "-l:libserverdbginfo.a"
  "-l:libtransport.a"
  "-l:libcommon.a"
  "-Wl,--end-group"
  "-l:libwangle.a"
  "-l:libfizz.a"
  "-l:libxxhash.a"
)
THIRD_PARTY_LDFLAGS+="${THRIFT_SERVICE_LDFLAGS[*]} "
THIRD_PARTY_LDFLAGS+="$(pkg-config --libs --static libfolly) "
THIRD_PARTY_LDFLAGS+="-l:libglog.a -l:libgflags.a -l:libboost_context.a -l:libfmt.a -l:libssl.a -l:libcrypto.a"

echo "$THIRD_PARTY_LDFLAGS"

if [[ -z "${NVCC_GENCODE-}" ]]; then
    IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
    arch_gencode=""
    for arch in "${arch_array[@]}"
    do
        case "$arch" in
        "p100")
        arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
            ;;
        "v100")
        arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
            ;;
        "a100")
        arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
        ;;
        "h100")
            arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
        ;;
        "b200")
            arch_gencode="$arch_gencode -gencode=arch=compute_100,code=sm_100"
        ;;
        esac
    done
    NVCC_GENCODE=$arch_gencode
fi

if [ "$CLEAN_BUILD" == 1 ]; then
    rm -rf "$BUILDDIR"
fi

mkdir -p "$BUILDDIR"
pushd "${NCCL_HOME}"

function build_nccl {
  make VERBOSE=1 -j${BUILD_JOBS} \
    src.build \
    BUILDDIR="$BUILDDIR" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    NCCL_SUFFIX="x-${DEV_SIGNATURE}" \
    NCCL_FP8="$NCCL_FP8" \
    BASE_DIR="$BASE_DIR" \
    CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
    CONDA_LIB_DIR="$CONDA_LIB_DIR" \
    THIRD_PARTY_LDFLAGS="$THIRD_PARTY_LDFLAGS" \
    CUDARTLIB="$CUDARTLIB"
}

function build_and_install_nccl {
make VERBOSE=1 -j${BUILD_JOBS} \
    src.install \
    BUILDDIR="$BUILDDIR" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    NCCL_SUFFIX="x-${DEV_SIGNATURE}" \
    NCCL_FP8="$NCCL_FP8" \
    BASE_DIR="$BASE_DIR" \
    CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
    CONDA_LIB_DIR="$CONDA_LIB_DIR" \
    THIRD_PARTY_LDFLAGS="$THIRD_PARTY_LDFLAGS" \
    CUDARTLIB="$CUDARTLIB"
}

if [[ -z "${NCCL_BUILD_AND_INSTALL}" ]]; then
  build_nccl
else
  build_and_install_nccl
fi

# sanity check
if [ -n "${NCCL_RUN_SANITY_CHECK}" ]; then
    pushd examples
    export NCCL_DEBUG=WARN
    export LD_LIBRARY_PATH=$BUILDDIR/lib

    make all \
      NVCC_GENCODE="$NVCC_GENCODE" \
      CUDA_HOME="$CUDA_HOME" \
      NCCL_HOME="$INSTALL_PREFIX"

    set +e

    TIMEOUT=10s
    timeout $TIMEOUT "$BUILDDIR"/examples/HelloWorld
    if [ "$?" == "124" ]; then
        echo "Program TIMEOUT in ${TIMEOUT}. Terminate."
    fi
    popd
fi

popd
