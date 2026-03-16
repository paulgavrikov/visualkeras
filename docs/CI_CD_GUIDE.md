# CI/CD Documentation Guide

This guide explains the continuous integration and continuous deployment setup for the visualkeras documentation.

## Overview

The visualkeras project uses two systems for ensuring documentation quality:

1. **GitHub Actions** - Builds and validates docs on every push/PR
2. **ReadTheDocs** - Hosts and deploys docs to the web

Both systems work together to maintain high quality documentation.

## GitHub Actions Workflow

### What It Does

The workflow in `.github/workflows/build-docs.yml` automatically:

1. Builds Sphinx documentation on every push and pull request
2. Installs all dependencies (Sphinx, theme, packages)
3. Reports build status and warnings
4. Uploads build artifacts for review
5. Adds summary to workflow run page

### Triggers

The workflow runs when:

- You push to: master, camera-ready, or main
- You open a pull request against those branches
- Changes touch: docs/ or visualkeras/ directories

To disable, modify the `on:` section in `.github/workflows/build-docs.yml`.

### Workflow Steps

1. **Checkout code** - Download your repository
2. **Set up Python** - Install Python 3.11
3. **Cache pip** - Speed up dependency installation
4. **Install dependencies** - Install Sphinx and packages
5. **Build documentation** - Run sphinx-build
6. **Upload artifacts** - Save HTML output (7 day retention)
7. **Post summary** - Display results on GitHub

### Viewing Results

After pushing:

1. Go to GitHub repo
2. Click "Actions" tab
3. Find your workflow run
4. Click to view logs and status
5. Download "sphinx-html" artifact to preview locally

### Build Status Badge

To add a documentation build badge to your README:

```markdown
![Docs](https://github.com/paulgavrikov/visualkeras/actions/workflows/build-docs.yml/badge.svg)
```

This shows the current build status at a glance.

## ReadTheDocs Deployment

### What It Does

ReadTheDocs automatically:

1. Builds documentation on every push
2. Hosts it publicly at visualkeras.readthedocs.io
3. Maintains version history
4. Handles domain management
5. Provides search functionality

### Configuration

The `.readthedocs.yml` file in the project root tells RTD how to build:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
```

This specifies the build environment.

```yaml
python:
  version: 3.11
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

This installs dependencies and the visualkeras package itself.

For detailed setup, see [READTHEDOCS_SETUP.md](READTHEDOCS_SETUP.md).

## Connecting the Systems

### GitHub Actions for Quality Control

Use GitHub Actions to catch issues before merging:

1. Run on pull requests to validate changes
2. Fail the check if warnings are too high
3. Require passing checks before merge

Modify `.github/workflows/build-docs.yml` to enable strict mode:

```yaml
- name: Build documentation
  run: |
    cd docs
    sphinx-build -W -b html source build/html
```

The `-W` flag converts warnings to errors.

### ReadTheDocs for Production Hosting

ReadTheDocs handles the final public build:

1. Triggers after GitHub Actions passes
2. Builds for multiple Python versions (optional)
3. Hosts documentation publicly
4. Maintains documentation history

## Manual Triggers

### Rebuild on GitHub Actions

If you need to manually rebuild without code changes:

1. Go to Actions tab
2. Select "Build Documentation" workflow
3. Click "Run workflow"
4. Choose your branch
5. Click "Run workflow" button

### Rebuild on ReadTheDocs

If you need to manually rebuild RTD docs:

1. Visit readthedocs.org
2. Go to your project
3. Click "Builds" tab
4. Click "Build Version" button
5. Select your branch
6. Click "Build" or "Force Build"

## Monitoring

### Check Build Status

**GitHub Actions:**

1. Click Actions tab in GitHub
2. View latest run status
3. Click run to view details

**ReadTheDocs:**

1. Visit project dashboard at readthedocs.org
2. Check "Last Built" timestamp
3. Click on build to view logs

### Get Notifications

**GitHub Actions:**

Notifications happen by default in GitHub. Configure in:

- Settings > Notifications
- Select "Watching" for your repo

**ReadTheDocs:**

Configure in project Admin > Notifications:

- Email on build failures
- Slack/Discord integration
- Custom webhooks

## Common Tasks

### Update Documentation

1. Edit the relevant .rst files in docs/source/
2. Commit and push to GitHub
3. GitHub Actions automatically builds
4. ReadTheDocs builds once GitHub changes appear
5. Visit visualkeras.readthedocs.io to see updates (5-10 min delay)

### Fix Build Errors

When a build fails:

1. Check GitHub Actions logs for quick feedback
2. Check ReadTheDocs logs for full details
3. Fix the issue locally (see below)
4. Commit and push

### Test Locally

Before pushing, build and test locally:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html source build/html
python -m http.server 8000 --directory build/html
```

Open <http://localhost:8000> in your browser.

### Add a New Documentation Page

1. Create a new .rst file in docs/source/
2. Add it to a toctree in the appropriate index.rst
3. Build locally to test
4. Commit and push
5. CI/CD systems automatically detect and build it

## Troubleshooting

### GitHub Actions Build Fails

Check the GitHub Actions log:

1. Go to Actions tab
2. Click on failed run
3. Expand "Build documentation" step
4. Look for error messages

Common issues:

- Missing dependencies in docs/requirements.txt
- Python import errors (check conf.py)
- RST syntax errors in documentation

### ReadTheDocs Build Fails

Check the ReadTheDocs log:

1. Go to readthedocs.org
2. Click on your project
3. Go to Builds tab
4. Click on failed build
5. Check the build log

Common issues:

- Core package dependencies not installed
- Incompatible Python version
- Theme not installed

### Different Results on GitHub vs ReadTheDocs

This can happen if environments differ. To match:

1. Ensure Python version matches in both
2. Pin exact versions in docs/requirements.txt
3. Test locally with both builds
4. Check both config files for differences

## Files Referenced

- `.github/workflows/build-docs.yml` - GitHub Actions workflow
- `.readthedocs.yml` - ReadTheDocs configuration
- `docs/source/conf.py` - Sphinx configuration
- `docs/requirements.txt` - Documentation dependencies

## Best Practices

1. **Test locally first** - Build docs locally before pushing
2. **Write clear commit messages** - Makes it easy to see what changed
3. **Group doc changes with code changes** - Keep related changes together
4. **Review automated reports** - Check GitHub Actions output
5. **Update .readthedocs.yml when dependencies change** - Keep config in sync
6. **Monitor build times** - Keep documentation builds fast

## Additional Resources

- GitHub Actions docs: <https://github.com/features/actions>
- ReadTheDocs docs: <https://docs.readthedocs.io/>
- Sphinx docs: <https://www.sphinx-doc.org/>
