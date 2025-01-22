{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    flake-utils,
    nixpkgs
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in rec {
        devShells.default = with pkgs;
          mkShell rec {
            nativeBuildInputs = [
              cargo-fuzz
              linuxPackages_latest.perf
            ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [stdenv.cc.cc];
          };
      }
    );
}
