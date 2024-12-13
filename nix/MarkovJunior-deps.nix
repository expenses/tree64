{fetchNuGet}: [
  #(fetchNuGet { pname = "Microsoft.AspNetCore.App.Ref"; version = "7.0.20"; hash = "sha256-OEDXXjQ1HDRPiA4Y1zPr1xUeH6wlzTCJpts+DZL61wI="; })
  #(fetchNuGet { pname = "Microsoft.NETCore.App.Host.linux-x64"; version = "7.0.20"; hash = "sha256-Y1Dg8Sqhya86xD+9aJOuznT4mJUyFmoF/YZc0+5LBdc="; })
  #(fetchNuGet { pname = "Microsoft.NETCore.App.Ref"; version = "7.0.20"; hash = "sha256-W9RU3bja4BQLAbsaIhANQPJJh6DycDiBR+WZ3mK6Zrs="; })
  (fetchNuGet {
    pname = "Microsoft.NETCore.Platforms";
    version = "5.0.0";
    hash = "sha256-LIcg1StDcQLPOABp4JRXIs837d7z0ia6+++3SF3jl1c=";
  })
  (fetchNuGet {
    pname = "SixLabors.ImageSharp";
    version = "2.1.3";
    hash = "sha256-6f6WuuRdT7Lzbt80o9WC/B1R/DH5eYQR3yFtsU8GC4s=";
  })
  (fetchNuGet {
    pname = "System.Runtime.CompilerServices.Unsafe";
    version = "5.0.0";
    hash = "sha256-neARSpLPUzPxEKhJRwoBzhPxK+cKIitLx7WBYncsYgo=";
  })
  (fetchNuGet {
    pname = "System.Text.Encoding.CodePages";
    version = "5.0.0";
    hash = "sha256-YJ5jJqkVPp+6fEzSXOmw1sNSdygB5Rx7TJ0TrNS/wq4=";
  })
]
