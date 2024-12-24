{
  mkWindowsApp,
  wineWowPackages,
  fetchurl,
  version ? "0.99.7.1",
}: let
  srcs = {
    "0.99.7.1" = fetchurl {
      url = "https://github.com/ephtracy/ephtracy.github.io/releases/download/0.99.7/MagicaVoxel-0.99.7.1-win64.zip";
      hash = "sha256-z/Ia4EYi8LGmSzPGbxP+UHbzACrKvR1y9MSTmn3x4d4=";
    };
    "0.99.6.4" = fetchurl {
      url = "https://github.com/ephtracy/ephtracy.github.io/releases/download/0.99.6/MagicaVoxel-0.99.6.4-win64.zip";
      hash = "sha256-/StkFBzF4zoBzoBwCKNzm/PIbDDnj7Gx+NhV6/9a+L0=";
    };
  };
in
  mkWindowsApp rec {
    inherit version;
    wine = wineWowPackages.base;
    pname = "magicavoxel";
    name = "MagicaVoxel";
    enableMonoBootPrompt = false;
    wineArch = "win64";
    src = srcs."${version}";
    dontUnpack = true;
    winAppInstall = ''
      unzip ${src} -d "$WINEPREFIX"
    '';
    winAppRun = ''wine "$WINEPREFIX/MagicaVoxel-${version}-win64/MagicaVoxel.exe"'';
    installPhase = ''
      runHook preInstall

      ln -s $out/bin/.launcher $out/bin/${pname}

      runHook postInstall
    '';
  }
