// Builds csrc/cxshim.c against the system UCX.
// UCX prefix: $UCX_HOME, else /opt/hpcx/ucx (HPC-X, also used by ucx-py on
// these machines), else /usr.
use std::path::PathBuf;

fn main() {
    let home = std::env::var("UCX_HOME").ok().map(PathBuf::from).unwrap_or_else(|| {
        let hpcx = PathBuf::from("/opt/hpcx/ucx");
        if hpcx.join("include/ucp/api/ucp.h").exists() {
            hpcx
        } else {
            PathBuf::from("/usr")
        }
    });
    let lib = home.join("lib");
    cc::Build::new()
        .file("csrc/cxshim.c")
        .include(home.join("include"))
        .opt_level(2)
        .warnings(true)
        .compile("cxshim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    // rpath: binaries and the Python module find libucp without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rustc-link-lib=dylib=ucp");
    println!("cargo:rustc-link-lib=dylib=ucs");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rerun-if-changed=csrc/cxshim.c");
    println!("cargo:rerun-if-env-changed=UCX_HOME");
}
