use std::env;
use std::path::PathBuf;
use std::process::Command;

const PYMATCHING_REPO: &str = "https://github.com/oscarhiggott/PyMatching.git";
const PYMATCHING_REV: &str = "v2.4.0";

fn main() {
    println!("cargo:rerun-if-env-changed=NPSIM_PYMATCHING_SOURCE_DIR");
    println!("cargo:rerun-if-changed=native/pymatching_shim.cc");
    println!("cargo:rerun-if-changed=native/pymatching_shim.h");

    let source_dir = pymatching_source_dir();
    let source_root = source_dir.join("src");
    if !source_root
        .join("pymatching/sparse_blossom/matcher/mwpm.h")
        .exists()
    {
        panic!(
            "PyMatching source directory {:?} does not look like a PyMatching checkout",
            source_dir
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .include(&source_root)
        .include("native")
        .file("native/pymatching_shim.cc");

    for relative in PYMATCHING_SOURCES {
        build.file(source_root.join(relative));
    }

    build.flag_if_supported("-O3");
    build.flag_if_supported("-fPIC");
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        build.flag_if_supported("-mno-avx2");
    }
    build.compile("npsim_pymatching_sparse_blossom");
}

fn pymatching_source_dir() -> PathBuf {
    if let Ok(path) = env::var("NPSIM_PYMATCHING_SOURCE_DIR") {
        return PathBuf::from(path);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let checkout = out_dir.join("PyMatching");
    if checkout.join(".git").exists() {
        run(Command::new("git")
            .arg("-C")
            .arg(&checkout)
            .arg("checkout")
            .arg(PYMATCHING_REV));
        return checkout;
    }

    if checkout.exists() {
        panic!(
            "PyMatching checkout path {:?} exists but is not a git checkout",
            checkout
        );
    }
    run(Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg("--branch")
        .arg(PYMATCHING_REV)
        .arg(PYMATCHING_REPO)
        .arg(&checkout));
    checkout
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run {:?}: {err}", command));
    if !status.success() {
        panic!("command {:?} failed with status {status}", command);
    }
}

const PYMATCHING_SOURCES: &[&str] = &[
    "pymatching/sparse_blossom/flooder/graph.cc",
    "pymatching/sparse_blossom/flooder/detector_node.cc",
    "pymatching/sparse_blossom/flooder_matcher_interop/compressed_edge.cc",
    "pymatching/sparse_blossom/flooder/graph_fill_region.cc",
    "pymatching/sparse_blossom/flooder/match.cc",
    "pymatching/sparse_blossom/flooder/graph_flooder.cc",
    "pymatching/sparse_blossom/matcher/alternating_tree.cc",
    "pymatching/sparse_blossom/matcher/mwpm.cc",
    "pymatching/sparse_blossom/flooder_matcher_interop/region_edge.cc",
    "pymatching/sparse_blossom/flooder_matcher_interop/mwpm_event.cc",
    "pymatching/sparse_blossom/tracker/flood_check_event.cc",
    "pymatching/sparse_blossom/search/search_graph.cc",
    "pymatching/sparse_blossom/search/search_detector_node.cc",
    "pymatching/sparse_blossom/search/search_flooder.cc",
];
