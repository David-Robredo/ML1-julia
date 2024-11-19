## Using in Julia file or Jupyter Notebook

One can directly use this package by adding this Github repository as a package.

```
import Pkg;
Pkg.add(url="https://github.com/AndyBarcia/ML1-julia.git");
using ML1julia;
```

Otherwise, one may clone the repository, and add the local package.

```
git clone https://github.com/AndyBarcia/ML1-julia.git
Pkg.add(path="./ML1-julia");
using ML1julia;
```

# Using in REPL

For development, inside the REPL you can press `]` and call `activate .` to activate the project defined in the folder we are on. You can use `precompile` to force the compilation of the dependencies and the package, though it will be precompiled later if you use it.

```
pkg> activate .
(ML1julia) pkg> precompile
```

Once the environment is activated, you can return to the REPL and write `using ML1julia` to start using any of the functions defined in the module.

```
julia> holdOut(5, 0.5)
([4,3,2], [1,5])
```

Note: to allow recompiling the file, you can use `using Revise` when first starting julia. This module allows automatically compiling the package every time it is changed.

## Executing tests

Inside the activated environment, you can execute all tests with a call to `test`.

```
(ML1julia) pkg> test ML1julia
```