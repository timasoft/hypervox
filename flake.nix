{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; overlays = [ rust-overlay.overlays.default ]; };

      rust = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets    = [ "wasm32-unknown-unknown" ];
      };
    in {
      packages.ndvoxgcalc = pkgs.stdenv.mkDerivation {
        pname = "ndvoxgcalc";
        version = "0.1.0";
        src = ./.;
        nativeBuildInputs = [ rust pkgs.trunk pkgs.binaryen pkgs.pkg-config ];

        buildPhase = "trunk build --release --public-url ./";
        installPhase = ''
          mkdir -p $out
          cp -r dist/* $out/
        '';
      };

      defaultPackage = self.packages.${system}.ndvoxgcalc;

      devShells.default = pkgs.mkShell {
        buildInputs = [
          pkgs.fish
          rust
          pkgs.trunk pkgs.wasm-bindgen-cli pkgs.pkg-config
        ];

        shellHook = ''
          if [ -z "$FISH_VERSION" ] && [ -z "$NO_AUTO_FISH" ]; then
            exec ${pkgs.fish}/bin/fish
          fi
        '';

        env.RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
      };
    });
}
