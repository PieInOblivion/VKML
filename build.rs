use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Tell cargo to rerun this script if shaders change
    println!("cargo:rerun-if-changed=src/shaders/src");

    // Set up shader directories
    let shader_src_dir = PathBuf::from("src/shaders/src");
    let shader_out_dir = PathBuf::from("src/shaders");

    // Compile all compute shaders in the source directory
    compile_shaders(&shader_src_dir, &shader_out_dir);
}

fn compile_shaders(src_dir: &Path, out_dir: &Path) {
    // Read all files in the source directory
    let entries = fs::read_dir(src_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "comp")
                .unwrap_or(false)
        });

    for entry in entries {
        let src_path = entry.path();
        let file_name = src_path.file_name().unwrap().to_str().unwrap();
        let out_path = out_dir.join(file_name.replace(".comp", ".spv"));

        // Compile the shader using glslc
        let output = Command::new("glslc")
            .arg("--target-env=vulkan1.0")
            .arg("-fshader-stage=compute")
            .arg("-o")
            .arg(&out_path)
            .arg(&src_path)
            .output()
            .expect("Failed to execute glslc");

        if !output.status.success() {
            panic!(
                "Failed to compile shader {}:\n{}",
                file_name,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        println!("Successfully compiled: {}", file_name);
    }
}
