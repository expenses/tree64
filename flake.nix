{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.crane.url = "github:ipetkov/crane";
  inputs.erosanix.url = "github:emmanuelrosa/erosanix";
  inputs.nixpkgs-slang.url = "github:expenses/nixpkgs/shader-slang";

  inputs.fenix = {
    url = "github:nix-community/fenix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
    erosanix,
    fenix,
    nixpkgs-slang,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-slang = nixpkgs-slang.legacyPackages.${system};
        craneLib = crane.mkLib pkgs;
        python-deps = ps: [ps.ipython ps.numpy ps.pillow ps.scikit-image ps.ffmpeg-python ps.zstandard ps.openusd];
      in rec {
        packages = rec {
          rga = with pkgs;
            stdenv.mkDerivation {
              name = "rga";
              src = fetchzip {
                url = "https://gpuopen.com/download/radeon-developer-tool-suite/RadeonDeveloperToolSuite-2024-09-26-1411.tgz";
                hash = "sha256-e1nu0afKdxUZDyQn7OHD+8IQhw+nIedud7LlIs52F1M=";
              };
              installPhase = ''
                mkdir $out
                cp -r $src $out/bin
              '';
              buildInputs = [libGL stdenv.cc.cc zlib libxml2 libxkbcommon libpng fontconfig xorg.libSM libdrm wayland];
              nativeBuildInputs = [
                autoPatchelfHook
              ];
            };

          magicavoxel = pkgs.callPackage ./nix/magicavoxel.nix {
            inherit (erosanix.lib.${system}) mkWindowsApp;
          };
          # Mostly useful as it writes files out in version 150, not the undocumented 200.
          magicavoxel-old = pkgs.callPackage ./nix/magicavoxel.nix {
            version = "0.99.6.4";
            inherit (erosanix.lib.${system}) mkWindowsApp;
          };
          avoyd = pkgs.callPackage ./nix/avoyd.nix {
            inherit (erosanix.lib.${system}) mkWindowsApp;
          };
          MarkovJunior = with pkgs;
            buildDotnetModule {
              src = ./MarkovJunior;
              name = "MarkovJunior";
              dotnet-sdk = dotnetCorePackages.sdk_7_0;
              dotnet-runtime = dotnetCorePackages.runtime_7_0;
              nugetDeps = ./nix/MarkovJunior-deps.nix;
            };
          wheel = craneLib.buildPackage rec {
            src = pkgs.lib.cleanSource ./.;
            cargoArtifacts = craneLib.buildDepsOnly {inherit src nativeBuildInputs;};
            buildPhaseCargoCommand = "maturin build --release";
            installPhaseCommand = "mv target/wheels $out";
            doCheck = false;
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
                #self."Pillow"
                #self."imageio"
                #self."numpy"
              ];
            };
          patched-python = pkgs.python3.override {
            packageOverrides = self: super: {
              markov = python-lib;
              inherit voxypy;
            };
          };
          usd2gltf = with pkgs;
            writeShellScriptBin
            "usd2gltf"
            "${blender}/bin/blender --background -P ${./nix/convert.py} -- $@";
          format = with pkgs;
            writeShellScriptBin "format" ''
              ${alejandra}/bin/alejandra . &&
              ${black}/bin/black . &&
              cargo fmt
            '';
          gen-flamegraph = with pkgs; writeShellScriptBin "gen-flamegraph" "${linuxPackages_latest.perf}/bin/perf script | ${flamegraph}/bin/stackcollapse-perf.pl | ${flamegraph}/bin/flamegraph.pl > out.svg";
          wasm-bindgen-99 = pkgs.wasm-bindgen-cli.override {
            version = "0.2.99";
            hash = "sha256-1AN2E9t/lZhbXdVznhTcniy+7ZzlaEp/gwLEAucs6EA=";
            cargoHash = "sha256-DbwAh8RJtW38LJp+J9Ht8fAROK9OabaJ85D9C/Vkve4=";
          };

          voxviewer-wasm = let
            craneLib = (crane.mkLib pkgs).overrideToolchain (with fenix.packages.${system};
              combine [
                stable.rustc
                stable.cargo
                targets.wasm32-unknown-unknown.stable.rust-std
              ]);
          in
            craneLib.buildPackage rec {
              name = "voxviewer";
              src = pkgs.lib.cleanSource ./.;
              cargoArtifacts = craneLib.buildDepsOnly {
                name = "voxviewer-deps";
                inherit src CARGO_BUILD_TARGET cargoExtraArgs;
              };
              CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
              cargoExtraArgs = "--package=voxviewer";
              doCheck = false;
            };

          voxviewer-dir = with pkgs;
            runCommand "voxviewer-dir" {} ''
              ${wasm-bindgen-99}/bin/wasm-bindgen ${voxviewer-wasm}/bin/voxviewer.wasm --target web --out-dir $out
              cp -r ${./voxviewer/assets}/* $out
            '';

          serve-voxviewer = with pkgs; writeShellScriptBin "serve-voxviewer" "${caddy}/bin/caddy file-server --listen ':8000' --root ${voxviewer-dir}";
        };
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [
              black
              linuxPackages_latest.perf
              flamegraph
              (packages.patched-python.withPackages (ps: [ps.markov ps.voxypy] ++ (python-deps ps)))
            ];
          };
        devShells.build = with pkgs;
          mkShell {
            buildInputs =
              [
                python3
                python3.pkgs.pip
                maturin
                linuxPackages_latest.perf
                packages.gen-flamegraph
                black
              ]
              ++ (python-deps python3.pkgs);
            shellHook = ''
              # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
              # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
              export PIP_PREFIX=$(pwd)/_build/pip_packages
              export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
              export PATH="$PIP_PREFIX/bin:$PATH"
              unset SOURCE_DATE_EPOCH
            '';
          };
        devShells.venv =
          (pkgs.buildFHSUserEnv {
            name = "pipzone";
            targetPkgs = pkgs: (with pkgs; [
              python3
              python3Packages.pip
              python3Packages.virtualenv
              zlib
              maturin
              black
              linuxPackages_latest.perf
              packages.gen-flamegraph
              openusd
            ]);
            runScript = "bash -c '. env/bin/activate && bash'";
          })
          .env;
        devShells.tools = with pkgs;
          mkShell {
            buildInputs = [
              packages.usd2gltf
              openusd
            ];
          };
        devShells.bevy = with pkgs;
          mkShell rec {
            nativeBuildInputs = [
              (pkgs-slang.shader-slang.overrideAttrs (attrs: {
                src = fetchFromGitHub {
                  owner = "shader-slang";
                  repo = "slang";
                  rev = "a985e240a6d9edb1545e357d20805bf81fba975a";
                  hash = "sha256-H/ePYu6o926M22zussW1f15iYRJCq29TeNJzBD0eAao=";
                  fetchSubmodules = true;
                };
              }))
              pkg-config
              renderdoc
              spirv-tools
            ];
            buildInputs = [
              udev
              alsa-lib
              vulkan-loader
              xorg.libX11
              xorg.libXcursor
              xorg.libXi
              xorg.libXrandr # To use the x11 feature
              libxkbcommon
              wayland # To use the wayland feature
            ];
            LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          };
      }
    );
}
