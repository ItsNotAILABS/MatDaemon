# Calling MatDaemon from GitHub

MatDaemon includes a repository-local GitHub Action and a manual workflow so benchmark jobs can be launched from GitHub.

## Manual Run

Open **Actions -> matdaemon-benchmark -> Run workflow** and choose:

- `profile`: `quick`, `launch`, or `ai`
- `backends`: for example `numpy tiled`
- `repetitions`: benchmark repetitions per case
- `strict`: fail the run if any requested backend errors

The workflow uploads:

- `benchmark-results.json`
- `benchmark-results.md`

The Markdown report is also appended to the GitHub Actions step summary.

## Reuse from Another Workflow in This Repo

```yaml
jobs:
  matdaemon:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/matdaemon-benchmark
        with:
          profile: ai
          backends: numpy tiled
          repetitions: "1"
          strict: "true"
      - uses: actions/upload-artifact@v4
        with:
          name: matdaemon-benchmark-results
          path: matdaemon-benchmark-results
```

## Reuse from Another Repository

```yaml
- uses: ItsNotAILABS/MatDaemon/.github/actions/matdaemon-benchmark@main
  with:
    profile: quick
    backends: numpy tiled
    repetitions: "1"
```

This gives agents, maintainers, and CI systems a direct GitHub-native way to call MatDaemon and collect proof artifacts.
