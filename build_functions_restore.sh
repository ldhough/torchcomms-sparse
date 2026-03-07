function build_fb_oss_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  # Check if source exists in THIRD_PARTY_SRC
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    cp -r "${THIRD_PARTY_SRC}/${library_name}" .
    # Fix permissions on copied git repository
    chmod -R u+w "${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  local source_dir="../${library_name}/${library_name}"
  if [ -f ${library_name}/CMakeLists.txt ]; then
    source_dir="../${library_name}"
  fi
  if [ -f ${library_name}/build/cmake/CMakeLists.txt ]; then
    source_dir="../${library_name}/build/cmake"
  fi
  if [ -f ${library_name}/cmake_unofficial/CMakeLists.txt ]; then
    source_dir="../${library_name}/cmake_unofficial"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  if [[ "${CLEAN_THIRD_PARTY}" == 1 || "${INCREMENTAL_BUILD}" == 0 ]]; then
    rm -rf build-output
  fi
  mkdir -p build-output
  pushd build-output
  do_cmake_build "$source_dir" "$extra_flags"
  popd
}

function build_automake_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  # Check if source exists in THIRD_PARTY_SRC
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    cp -r "${THIRD_PARTY_SRC}/${library_name}" .
    # Fix permissions on copied git repository
    chmod -R u+w "${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
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

  # Check if source exists in THIRD_PARTY_SRC
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    cp -r "${THIRD_PARTY_SRC}/${library_name}" .
    # Fix permissions on copied git repository
    chmod -R u+w "${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
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

  # Check if source exists in THIRD_PARTY_SRC
  if [ -d "${THIRD_PARTY_SRC}/${library_name}" ]; then
    echo "Using pre-downloaded source for $library_name from ${THIRD_PARTY_SRC}"
    cp -r "${THIRD_PARTY_SRC}/${library_name}" .
    # Fix permissions on copied git repository
    chmod -R u+w "${library_name}"
  elif [ ! -e "$library_name" ]; then
    echo "WARNING: $library_name not found in ${THIRD_PARTY_SRC}, attempting git clone..."
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  pushd "$library_name"
  ./config no-shared --prefix="$INSTALL_PREFIX" --openssldir="$INSTALL_PREFIX" --libdir=lib

  make -j${BUILD_JOBS}
  make install
  popd
}