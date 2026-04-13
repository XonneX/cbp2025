{
  description = "Development flake for CPB2025";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      x86 = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages."${x86}";
    in
    {
      devShells."${x86}".default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          gcc
          gnumake
          zlib
          gdown
          python3
          python3Packages.pandas
          dos2unix
        ];

        # shellHook = 
        #   git config --local core.hooksPath .githooks/
        # ;

        # Environment Variables
        ENV_VAR = "";
      };
    };
}

