use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // rerun if anything under src/instruction changes
    println!("cargo:rerun-if-changed=src/instruction");

    // write compiled shaders to OUT_DIR/shaders
    // this requires every shader name be unique, which should be fine
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set")).join("shaders");
    fs::create_dir_all(&out_dir).expect("failed to create OUT_DIR/shaders");

    // iteratively walk src/instruction and compile any .comp files found
    let mut stack = vec![PathBuf::from("src/instruction")];

    while let Some(dir) = stack.pop() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }

                if path.extension().and_then(|s| s.to_str()) == Some("comp") {
                    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap();
                    let out_path = out_dir.join(file_name.replace(".comp", ".spv"));

                    let result = Command::new("glslc")
                        .arg("--target-env=vulkan1.0")
                        .arg("-fshader-stage=compute")
                        .arg("-o")
                        .arg(&out_path)
                        .arg(&path)
                        .output();

                    match result {
                        Ok(o) if o.status.success() => {
                            println!(
                                "Compiled shader: {} -> {}",
                                path.display(),
                                out_path.display()
                            );
                        }
                        Ok(o) => panic!(
                            "glslc failed for {}: {}",
                            path.display(),
                            String::from_utf8_lossy(&o.stderr)
                        ),
                        Err(e) => panic!(
                            "failed to run glslc for {}: {}. Install glslc or the Vulkan SDK.",
                            path.display(),
                            e
                        ),
                    }
                }
            }
        }
    }
}
