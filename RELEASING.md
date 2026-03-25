# Releasing

## One-time PyPI setup

1. Go to https://pypi.org/manage/account/publishing/
2. Add pending publisher:
   - Project: `turboquant-torch`
   - Owner: `nxank4`
   - Repo: `turboquant-torch`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Repeat for TestPyPI at https://test.pypi.org/manage/account/publishing/
   - Environment: `testpypi`
4. Create environments in GitHub repo settings:
   - Settings > Environments > New: `pypi` (require approval)
   - Settings > Environments > New: `testpypi`

## Publishing a release

1. Update version in `pyproject.toml`
2. Commit: `git commit -am "release: v0.1.0"`
3. Tag: `git tag v0.1.0`
4. Push: `git push origin main --tags`
5. CI runs tests, then publishes to TestPyPI, then PyPI
