# flake.nix
{
  description = "Python dev environment with pip";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            python
            python.pkgs.pip  # makes pip directly available
          ];

          # Optional: make sure pip installs go to a local directory, not global
          shellHook = ''
            export PIP_PREFIX="$PWD/.venv"
            export PYTHONPATH="$PIP_PREFIX/lib/python3.11/site-packages:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            echo "🔧 pip will install to: $PIP_PREFIX"
          '';
        };
      }
    );
}

