{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, crane, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };

      rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets = [ "wasm32-unknown-unknown" ];
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain (p: rustToolchain);

      runtimeDeps = with pkgs; [
        libX11
        libXcursor
        libXrandr
        libXi
        libXext
        wayland
        libxkbcommon
        libGL
        vulkan-loader
        fontconfig
        freetype
        alsa-lib
        udev
      ];

      libPath = pkgs.lib.makeLibraryPath runtimeDeps;

      pkgConfigPath = pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" runtimeDeps;

      src = pkgs.lib.fileset.toSource {
        root = ./.;
        fileset = pkgs.lib.fileset.unions [
          ./Cargo.toml ./Cargo.lock ./Trunk.toml ./index.html ./src
          (pkgs.lib.fileset.maybeMissing ./.cargo)
          (pkgs.lib.fileset.maybeMissing ./assets)
        ];
      };

      nativeCargoArtifacts = craneLib.buildDepsOnly {
        inherit src;
        nativeBuildInputs = with pkgs; [ pkg-config ];
        buildInputs = runtimeDeps;

        env = {
          PKG_CONFIG_PATH = pkgConfigPath;
          CARGO_PROFILE_RELEASE_LTO = "true";
          CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
        };
      };

      native = craneLib.buildPackage {
        pname = "ndvoxgcalc";
        version = "0.1.0";
        inherit src;
        cargoArtifacts = nativeCargoArtifacts;

        buildInputs = runtimeDeps;
        nativeBuildInputs = with pkgs; [ pkg-config makeWrapper ];

        env = {
          PKG_CONFIG_PATH = pkgConfigPath;
          CARGO_PROFILE_RELEASE_LTO = "true";
          CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
        };

        postFixup = ''
          if [ -x "$out/bin/ndvoxgcalc" ]; then
            wrapProgram "$out/bin/ndvoxgcalc" \
              --prefix LD_LIBRARY_PATH : "${libPath}" \
              --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib
          fi
        '';
      };

      webCargoArtifacts = craneLib.buildDepsOnly {
        inherit src;
        CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        doCheck = false;
        CARGO_PROFILE_RELEASE_LTO = "true";
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
      };

      web = craneLib.buildTrunkPackage {
        pname = "ndvoxgcalc-web";
        version = "0.1.0";
        inherit src;
        cargoArtifacts = webCargoArtifacts;

        CARGO_BUILD_TARGET = "wasm32-unknown-unknown";

        wasm-bindgen-cli = pkgs.wasm-bindgen-cli_0_2_117;

        nativeBuildInputs = with pkgs; [ trunk binaryen lld pkg-config ];

        CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER = "${pkgs.lld}/bin/lld";

        preBuild = ''
          export WASM_BINDGEN_CACHE_DIR="$TMPDIR/wasm-bindgen"
          export XDG_CACHE_HOME="$TMPDIR/.cache"
          mkdir -p "$WASM_BINDGEN_CACHE_DIR" "$XDG_CACHE_HOME"
        '';

        postInstall = ''
          cat > "$out/_headers" <<'EOF'
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
EOF
        '';

        CARGO_PROFILE_RELEASE_LTO = "true";
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
      };
    in
    {
      packages = {
        inherit native web;
        default = native;
      };

      apps = {
        native = flake-utils.lib.mkApp {
          drv = native;
          exePath = "/bin/ndvoxgcalc";
        };
        default = self.apps.${system}.native;
      };

      devShells.default = craneLib.devShell {
        inputsFrom = [ native web ];
        packages = with pkgs; [
          cargo rustc rustfmt clippy rust-analyzer
          trunk wasm-bindgen-cli binaryen vulkan-tools lld
          fish
        ];

        env = {
          LD_LIBRARY_PATH = "${libPath}:/run/opengl-driver/lib";
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER = "${pkgs.lld}/bin/lld";
        };

        shellHook = ''
          if [ -z "$FISH_VERSION" ] && [ -z "$NO_AUTO_FISH" ]; then
            exec ${pkgs.fish}/bin/fish
          fi
        '';
      };
    });
}
