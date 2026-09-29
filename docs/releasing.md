# TestPyPI rehearsal

This is a maintainer procedure for the first TestPyPI upload of the pre-alpha
`0.0.1.dev0` build. It does not certify the native-operation release contract or
authorize a production PyPI upload. The [design's release gate](design.md)
and [patch manifest](dev/xarray-upstream/xarray_patches.md) describe the current stock and
patched xarray evidence. In particular, stock xarray still has known paths that return a
valid-looking incorrect binding; a pre-alpha label does not make those paths safe.

## One-time hosted setup

Before any upload, confirm that publishing the package source is intended: TestPyPI artifacts
are public even when their GitHub repository is private. The repository and README links must
be publicly accessible before presenting the package as a community release. Resolve any
GitHub Actions billing or spending-limit failure so the validation job can start.

1. Create a separate account on [TestPyPI](https://test.pypi.org/). A production PyPI account
   is separate and does not log in to TestPyPI.
2. In the GitHub repository `matthiasschabel/xarrayrf`, create an environment named exactly
   `testpypi`. Set its deployment branch policy to **selected branches** and allow only `main`.
   Add required reviewers if available on the repository's GitHub plan and desired for this
   rehearsal. The workflow also skips jobs unless its ref is `main`; the environment restriction
   independently prevents dispatching feature-branch code with publishing credentials.
   Private-repository environments require a supported paid GitHub plan; required-reviewer
   availability has further plan restrictions. If using required reviewers as a solo maintainer,
   leave **Prevent self-review** off or arrange another reviewer.
3. In TestPyPI, add a **pending trusted publisher** for a new project with these exact fields:
   project name `xarrayrf`, GitHub owner `matthiasschabel`, repository `xarrayrf`, workflow
   filename `publish.yml`, environment `testpypi`. Use TestPyPI's pending-publisher form, not a
   PyPI API token. A pending publisher creates the project on its first successful OIDC upload;
   it does **not** reserve the name beforehand. Confirm that `xarrayrf` is still available on
   TestPyPI immediately before setup.

## Ordered rehearsal

1. On the intended `main` revision, run `uv sync --locked --extra dev`, `make check`,
   `make test`, and `make build`. Keep local `dist/` limited to the intended version's wheel and
   sdist; move older artifacts elsewhere before checking. `make build` checks their payloads.
   Run `uvx --from twine==7.0.0 twine check --strict dist/*`.
2. Review the workflow and hosted environment settings. Dispatch **TestPyPI rehearsal** from
   GitHub Actions with branch `main`. The build job runs the locked stock-xarray checks, builds
   wheel and sdist, validates metadata, rebuilds a wheel from the sdist, and installs the wheel
   into a clean environment outside the checkout for a core/native smoke test. The publishing
   job downloads only the build job's artifact and uses OIDC with the `testpypi` environment.
3. Inspect the new project and files on TestPyPI. In a fresh Python 3.12+ environment, install
   the core dependencies from PyPI and the exact rehearsal version from TestPyPI:

   ```sh
   python -m pip install --index-url https://pypi.org/simple/ 'numpy>=1.26' 'xarray>=2026.7.0'
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps 'xarrayrf==0.0.1.dev0'
   python -m pip check
   python -c 'from importlib.metadata import version; print(version("xarrayrf"))'
   ```

   Replace the version for later rehearsals. Run the smoke path from the workflow outside
   the source checkout. Verify the package page's metadata, README links, license and
   changelog in the sdist.

The published wheel declares ordinary `numpy` and `xarray` dependencies. The `tool.uv.sources`
pin for the optional patched development group does not become a wheel dependency, and TestPyPI
users receive stock xarray unless they explicitly install another build. Both stock and patched
lanes must satisfy the release gate before a production release; previous local patched-lane
results are recorded in the patch manifest, but this workflow does not run that lane.

Hosted setup progress is recorded in the [setup note](dev/architecture/release_and_repository_notes.md).
OIDC upload, TestPyPI installation, and production publishing remain unvalidated.
Changed artifacts need a new version because an
uploaded filename cannot be replaced. An unchanged artifact need not be uploaded again;
the workflow deliberately does not suppress duplicate-upload errors. Decide the next version
after reviewing the first rehearsal, and update `pyproject.toml` and `uv.lock` together.

References: [PyPA pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[trusted publisher setup](https://docs.pypi.org/trusted-publishers/using-a-publisher/), and
[TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/), and
[GitHub environment availability](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
