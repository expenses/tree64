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
          voxypy = with pkgs;
            python3Packages.buildPythonPackage rec {
              pname = "voxypy";
              version = "0.2.3";
              src = fetchurl {
                url = "https://files.pythonhosted.org/packages/52/1f/228a441d2ea6e5348eeb0e8076a55a017d75fe8007be411097f9bb91f652/voxypy-0.2.3-py3-none-any.whl";
                sha256 = "1icaa0q66q86cc2xf19l1cya8xbwylapkhk0blq9286wy6cwkjm1";
              };
              format = "wheel";
              doCheck = false;
              buildInputs = [];
              checkInputs = [];
              nativeBuildInputs = [];
              propagatedBuildInputs = [
                #self."Pillow"                                                                    self."imageio"                                                                   self."numpy"
              ];
            };
          patched-python = pkgs.python3.override {
            packageOverrides = self: super: {
              markov = python-lib;
              inherit voxypy;
            };
          };
          usd2gltf = with pkgs; writeShellScriptBin "usd2gltf" "${blender}/bin/blender --background -P ${./convert.py} -- $@";
        };
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [black tev (packages.patched-python.withPackages (ps: [ps.markov ps.voxypy ps.ipython ps.numpy ps.pillow ps.scikit-image ps.ffmpeg-python ps.zstandard ps.openusd]))];
          };
        devShells.build = with pkgs; mkShell {buildInputs = [python3];};
      }
    );
}
