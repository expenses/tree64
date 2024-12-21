{
  mkWindowsApp,
  wineWowPackages,
  fetchurl,
  requireFile
}: let
in
  mkWindowsApp rec {
    version = "0.24.0.929";
    wine = wineWowPackages.base;
    pname = "avoyd";
    name = "Avoyd";
    enableMonoBootPrompt = false;
    wineArch = "win64";
    src = requireFile {
      name = "Avoyd_${version}_demo_setup.exe";
      url = "https://www.enkisoftware.com/products";
      hash = "sha256-3W2lf9t6mBJq9W9DcsxJXGSp4XPasUZt5a0T8EQRU24=";
    };
    dontUnpack = true;
    winAppInstall = ''
      wine ${src}
    '';
    winAppRun = ''wine "$WINEPREFIX/drive_c/Program Files (x86)/Avoyd/bin64/Avoyd.exe"'';
    installPhase = ''
      runHook preInstall

      ln -s $out/bin/.launcher $out/bin/${pname}

      runHook postInstall
    '';
  }
