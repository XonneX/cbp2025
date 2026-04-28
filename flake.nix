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
      python = pkgs.python312;
      pythonPackages = python.pkgs;
      lib-path = with pkgs; lib.makeLibraryPath [
        libffi
        openssl
        stdenv.cc.cc
      ];
    in
    {
      devShells."${x86}".default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          stdenv.cc.cc.lib
          python
          pythonPackages.venvShellHook

          gcc
          gnumake
          zlib
          gdown
          dos2unix
        ];

        shellHook = ''
          SOURCE_DATE_EPOCH=$(date +%s)
          export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
          VENV=.venv

          if test ! -d $VENV; then
            python3.12 -m venv $VENV
          fi
          source ./$VENV/bin/activate
          export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
          pip install -r requirements.txt
        '';

        postShellHook = ''
          ln -sf ${python.sitePackages}/* ./.venv/lib/python3.12/site-packages
        '';
      };
    };
}

