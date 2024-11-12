{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.crane.url = "github:ipetkov/crane";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        craneLib = crane.mkLib pkgs;
      in rec {
        packages = rec {
          wheel = craneLib.buildPackage rec {
            src = craneLib.cleanCargoSource ./.;
            cargoArtifacts = craneLib.buildDepsOnly {inherit src nativeBuildInputs;};
            buildPhaseCargoCommand = "maturin build --release";
            installPhaseCommand = "mv target/wheels $out";
            nativeBuildInputs = [pkgs.python3 pkgs.maturin pkgs.rustc];
          };
          python-lib = with pkgs;
            python3Packages.buildPythonPackage {
              pname = "markov";
              version = "0.1.0";
              src = wheel;
              dontUnpack = true;
              buildPhase = ''
                cp -r $src dist
                chown $(whoami) -R dist
                chmod +w -R dist
              '';
            };
          patched-python = pkgs.python3.override {
            packageOverrides = self: super: {
              markov = python-lib;
            };
          };
        };
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [black tev (packages.patched-python.withPackages (ps: [ps.markov ps.numpy ps.ffmpeg-python ps.zstandard]))];
          };
      }
    );
}
